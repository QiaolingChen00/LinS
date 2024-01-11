import copy
import math

from simulator.ab_cost_model import allreduce, broadcast, p2p
from simulator.context import ParallelMode, check_and_modify_parallel_config
from simulator.context import global_context as gpc
from simulator.mem import (
    get_backward_mem_peak,
    get_block_output_mm,
    get_block_threshold,
    get_head_input_mm,
    get_head_output_mm,
    get_memory_pool_mm,
    get_norm_output_mm,
    get_p2p_buffer_size,
    get_rotary_emb_sincos_cache_mm,
)
from simulator.overlap import TransformerOverlapOneLayer

# from comm import TransformerCommunication
# from utils.utils import _get_model_config
from utils.common import _100GB, GB, AlgoType, cal_block_p_elem, get_model_config
from utils.config import Config


class LinsSolutionNoZ3:
    def __init__(
        self,
        pp,
        sp,
        wp,
        zp,
        seq_len,
        micro_bsz,
        micro_num,
        algo_type,
        pp_comm_cost,
        activation,
        zp_comm_cost,
        wp_comm_cost,
        sp_comm_cost,
        os_mm_cost,
        p_g_mm_cost,
        fwd_bwd_cost,
        mem_cost,
        comp_wp,
        comp_attn,
        world_size,
        activation_ckpt,
        tgs,
        mem_pool_mm,
        norm_activation,
        head_input_activation,
        head_output_activation,
        block_output_activation,
        wdp_comm_cost,
        all_fwd_bwd_cost,
        g_bsz,
        pp_p2p_buffer,
        rotary_emb_sincos_cache_mm,
        modelsize,
        backward_mem_peak,
        blocks_activation,
        overlap_latency,
        total_latency,
    ):
        self.pp = pp
        self.sp = sp
        self.seq_len = seq_len
        self.micro_bsz = micro_bsz
        self.micro_num = micro_num
        self.algo_type = algo_type
        self.pp_comm_cost = pp_comm_cost
        self.activation = activation
        self.activation_ckpt = activation_ckpt

        self.wp_size = wp
        self.zp_size = zp
        self.zp_comm_cost = zp_comm_cost
        self.wp_comm_cost = wp_comm_cost
        self.os_mm_cost = os_mm_cost
        self.p_g_mm_cost = p_g_mm_cost
        self.sp_comm_cost = sp_comm_cost
        self.total_mm_cost = mem_cost
        self.fwd_bwd_cost = fwd_bwd_cost
        self.comp_wp = comp_wp
        self.comp_attn = comp_attn
        self.world_size = world_size
        self.tgs = tgs

        self.mem_pool_mm = mem_pool_mm
        self.norm_activation = norm_activation
        self.head_input_activation = head_input_activation
        self.head_output_activation = head_output_activation
        self.block_output_activation = block_output_activation

        self.wdp_comm_cost = wdp_comm_cost
        self.all_fwd_bwd_cost = all_fwd_bwd_cost
        self.g_bsz = g_bsz
        self.pp_p2p_buffer = pp_p2p_buffer
        self.rotary_emb_sincos_cache_mm = rotary_emb_sincos_cache_mm
        self.modelsize = modelsize
        self.backward_mem_peak = backward_mem_peak
        self.blocks_activation = blocks_activation
        self.overlap_latency = overlap_latency
        self.total_latency = total_latency

    def __str__(self):
        return self.__repr__()

    # Begin: world_size: 128, pp:1, sp:16, micro_bsz:1, micro_num:2, algo_type:isp, wp:16, zp:4 ckpt:1
    def __repr__(self):
        return (
            f" world_size: {self.world_size}"
            f" tgs: {self.tgs}, total_latency:{self.total_latency*10**3:.3f} ms"
            f" global bsz: {self.g_bsz} \n"
            f" activation ckpt: {self.activation_ckpt}"
            f" seq_len: {self.seq_len}"
            f" micro_bsz: {self.micro_bsz}"
            f" micro_num: {self.micro_num}, \n"
            f" modelsize: {self.modelsize}, algo_type: {self.algo_type}, pp_size: {self.pp}, sp_size: {self.sp}, wp_size: {self.wp_size}, zp_size: {self.zp_size}, \n"
            f" one micro step fwd_bwd_cost: {self.fwd_bwd_cost*10**3:.2f} ms, all_fwd_bwd_cost: {self.all_fwd_bwd_cost*10**3:.2f} ms, overlap_latency: {self.overlap_latency*10**3:.2f} ms\n"
            f" COMP: comp_wp: {self.comp_wp*10**3:.2f} ms, comp_attn: {self.comp_attn*10**3:.2f} ms, \n"
            f" COMM: pp_comm_cost: {self.pp_comm_cost*10**3:.2f} ms, zp_comm_cost: {self.zp_comm_cost*10**3:.2f} ms, one layer wp_comm_cost: {self.wp_comm_cost*10**3:.2f} ms, one layer sp_comm_cost: {self.sp_comm_cost*10**3:.2f} ms, wdp_comm_cost: {self.wdp_comm_cost*10**3:.2f} ms \n"
            f" total mem_cost: {self.total_mm_cost /GB:.2f} GB \n"
            f" Not evictable MEM: os_mm_cost: {self.os_mm_cost/GB:.2f} GB, p_g_mm_cost: {self.p_g_mm_cost/GB:.2f} GB, isp_mem_pool: {self.mem_pool_mm/GB:.2f} GB, sincos_cache_mm: {self.rotary_emb_sincos_cache_mm/GB:.2f} GB,pp_p2p_buffer: {self.pp_p2p_buffer/GB:.2f} GB\n"
            f" Activation MEM: total activation: {self.activation/GB:.2f} GB, blocks_activation: {self.blocks_activation/GB:.2f} GB, norm_activation: {self.norm_activation/GB:.2f} GB,backward_mem_peak: {self.backward_mem_peak/GB:.2f} GB \n"
            f" head_input_activation: {self.head_input_activation/GB:.2f} GB, head_output_activation: {self.head_output_activation/GB:.2f} GB, block_output_activation(enable ckpt): {self.block_output_activation/GB:.2f} GB \n"
        )


class SPIter:
    def __init__(self, gpu_nums, head_nums):
        assert head_nums % 2 == 0
        stop = min(gpu_nums, head_nums)
        if gpu_nums <= 8:
            self.num_list = [1] + list(range(2, stop + 1, 2))
        else:
            self.num_list = [1] + list(range(2, 8, 2)) + list(range(8, stop + 1, 8))

    def __iter__(self):
        return iter(self.num_list)

    def __len__(self):
        return len(self.num_list)


class PPIter:
    def __init__(self, gpu_nums, layer_nums):
        assert layer_nums % 2 == 0
        stop = int(math.log2(min(gpu_nums, layer_nums)))
        self.num_list = [2**i for i in range(stop + 1)]

    def __iter__(self):
        return iter(self.num_list)

    def __len__(self):
        return len(self.num_list)


class Constraint:
    def __init__(
        self,
        overlap_wdp: int,
        debug: bool,
        config: dict,
        use_fixed_micro_bsz: bool,
        use_strict_bsz: bool,
    ) -> None:
        """求解器

        Args:
            world_size (int): GPU数量(现在这个参数没用)
            global_bsz (int): use_strict_bsz为True时会用到这个bsz
            global_bsz_min (int): global_bsz的搜素上界
            global_bsz_max (int): global_bsz的搜素下界
            max_world_size (int): world_size的搜素上界
            min_world_size (int): world_size的搜素下界
            seq_len (int):
            overlap_wdp (int): 是否考虑overlap wdp的通信
            fixed_micro_num (int): 是否固定micro_num,默认为None不生效
            fixed_micro_bsz (int): 是否固定micro_bsz ,默认为None不生效
            debug (bool): 是否输出额外的debug信息
            use_strict_bsz(bool) : 如果为True, 则会严格限制globa bsz为global_bsz参数的值
            config (dict): 模型的config
        """

        self.global_bsz = config.global_bsz  # 4
        self.global_bsz_min = config.global_bsz_min  # 4
        self.global_bsz_max = config.global_bsz_max  # 5
        self.max_world_size = config.world_size_max
        self.min_world_size = config.world_size_min
        self.fixed_micro_num = config.fixed_micro_num
        self.fixed_micro_bsz = config.fixed_micro_bsz
        self.use_fixed_micro_bsz = use_fixed_micro_bsz
        self.debug = debug
        self.overlap_wdp = overlap_wdp

        self.seq_len = config.sequence_length
        self.dtype_size = config.dtype_size
        self.model_size = config.model_size
        self.vocab_size = config.vocab_size
        self.use_fa = config.use_fa
        self.mem_threshold = config.mem_threshold
        self.fp32_ratio = max(1, 4 // self.dtype_size)
        # self._param_elements = float(self.model_size * 10**9)
        self._h, self._a, self._l, self.mlp_ratio, self.multiple_of, self._param_elements = get_model_config(
            self.model_size
        )
        self._algo_list = [AlgoType.ISP, AlgoType.MSP, AlgoType.FSP]
        self._use_strict_bsz = use_strict_bsz
        self._wp_penalty_coefficient = config.wp_penalty_coefficient
        assert 0 < self._wp_penalty_coefficient <= 1
        if self._use_strict_bsz:
            assert (
                not self.use_fixed_micro_bsz
            ), "If use 'use_fixed_micro_bsz', the solution satisfies 'use_strict_bsz' cannot be found."

        self.min_comm_cost, self.msp_min_cost, self.fsp_min_cost, self.isp_min_cost = (
            float("inf"),
            float("inf"),
            float("inf"),
            float("inf"),
        )
        self.min_cost_solution, self.msp_min_solu, self.fsp_min_solu, self.isp_min_solu = None, None, None, None

    def get_bsz_strict(self, world_size: int, pp_size: int, sp_size: int, seq_len: int):
        """
        严格的按照 global_bsz 限制返回满足要求的 micro_bsz 和 micro_num
        Args:
            pp_size (int)
            sp_size (int)
            seq_len (int)

        Returns:
            List[(int, int)]: micro_bsz, micro_num
        """
        if pp_size * sp_size > world_size:
            return None

        dp_world_size = world_size // pp_size // sp_size
        if world_size % pp_size != 0 or world_size % sp_size != 0 or world_size % (pp_size * sp_size) != 0:
            return None

        if self.global_bsz % dp_world_size != 0:
            return None
        if self.global_bsz % seq_len != 0:
            return None
        if self.global_bsz % (dp_world_size * seq_len) != 0:
            return None

        bsz = self.global_bsz // dp_world_size // seq_len

        micro_bsz_num = []
        for micro_bsz in range(1, bsz + 1):
            if bsz % micro_bsz == 0:
                micro_num = bsz // micro_bsz
                if micro_num >= pp_size:  # 我们暂时不考虑 micro_num < pp_size 的情况
                    micro_bsz_num.append((micro_bsz, micro_num))
        return micro_bsz_num

    def get_bsz_approximate(self, world_size: int, pp_size: int, sp_size: int, seq_len: int):
        """
        允许global bsz在 min_bsz 和 max_bsz 之间松弛
        Args:
            pp_size (int)
            sp_size (int)
            seq_len (int)

        Returns:
            List[(int, int)]: micro_bsz, micro_num
        """
        if pp_size * sp_size > world_size:
            return None

        dp_world_size = world_size // pp_size // sp_size
        if world_size % pp_size != 0 or world_size % sp_size != 0 or world_size % (pp_size * sp_size) != 0:
            return None

        bsz_max = self.global_bsz_max // dp_world_size // seq_len
        bsz_min = self.global_bsz_min // dp_world_size // seq_len

        micro_bsz_num = []
        for micro_bsz in range(1, int(bsz_max**0.5) + 1):
            for micro_num in range(1, int(bsz_max**0.5) + 1):
                if micro_bsz * micro_num >= bsz_min:
                    if micro_num >= pp_size:  # 我们暂时不考虑 micro_num < pp_size 的情况
                        assert micro_bsz * micro_num >= bsz_min and micro_bsz * micro_num <= bsz_max
                        micro_bsz_num.append((micro_bsz, micro_num))
        return micro_bsz_num

    @staticmethod
    def pp_bubble_fraction(pp_size, micro_num):
        return pp_size - 1 / micro_num

    def pp_comm_overhead(self, pp_size, sp_size, micro_bsz, micro_num):
        """计算pp中p2p通信的延迟"""
        if pp_size == 1:
            return 0

        p2p_buffer_size = get_p2p_buffer_size(self.dtype_size, self.seq_len, sp_size, micro_bsz, self._h)

        warmup_p2p_num = min(pp_size, micro_num)
        one_f_one_b_p2p_num = micro_num - 1
        cooldown_p2p_num = min(pp_size, micro_num)

        p2p_latency = (warmup_p2p_num + one_f_one_b_p2p_num + cooldown_p2p_num) * p2p(
            p2p_buffer_size, ParallelMode.PIPELINE
        )
        return p2p_latency

    def comm_dp_cost(self, algo, pp_blocks_elem, embedding_elem, zp) -> float:
        """切分OS引入的参数同步的通信开销"""
        # wdp引入的通信开销, 这里传入的是一个dp rank视角下的通信量
        # msp和fsp的参数被tp切了，需要除以sp_size
        # isp的参数被wp切了，需要除以wp_size
        if algo in [AlgoType.MSP, AlgoType.FSP]:
            # 同步梯度
            wdp_latency = allreduce(self.dtype_size * (pp_blocks_elem + embedding_elem), ParallelMode.DATA)

            # 同步参数
            zp_latency = zp * broadcast(self.dtype_size * (pp_blocks_elem + embedding_elem) / zp, ParallelMode.ZERO1)
        elif algo == AlgoType.ISP:
            # 同步梯度
            wdp_block_latency = allreduce(self.dtype_size * pp_blocks_elem, ParallelMode.WEIGHT_DATA)
            wdp_embedding_latency = allreduce(self.dtype_size * embedding_elem, ParallelMode.DATA)
            wdp_latency = wdp_block_latency + wdp_embedding_latency

            # 同步参数
            block_zp_latency = zp * broadcast(self.dtype_size * pp_blocks_elem / zp, ParallelMode.ZERO1)
            embedding_zp_latency = broadcast(self.dtype_size * embedding_elem, ParallelMode.DATA)
            zp_latency = max(block_zp_latency, embedding_zp_latency)

        return zp_latency, wdp_latency

    def build_parallel_config(self, algo_type: AlgoType, world_size, pp, sp, wp, zp):
        """TODO: add wdp,
        wdp不需要出现在config里,可以自动算出来

        """
        try:
            pipeline = dict(size=pp, interleaved_overlap=True)
            tensor = dict(size=sp, mode=algo_type)
            zero1 = dict(size=zp, fsdp=False)
            weight = dict(size=wp, overlap=True, memory_pool=True)

            parallel_conf = Config(
                {
                    "parallel": dict(
                        zero1=zero1,
                        tensor=tensor,
                        pipeline=pipeline,
                        weight=weight,
                    )
                }
            )

            gpc.load_config(parallel_conf)
            gpc.init_global_dist(0, world_size, "nccl", 1, 1)
            gpc.init_parallel_groups()  # work globally
            check_and_modify_parallel_config(parallel_conf)
        except AssertionError as e:
            if self.debug:
                print(f"NO solu: build gpc failed: {e}\n", flush=True)
            return None
        except ZeroDivisionError as e:
            if self.debug:
                print(f"NO solu: build gpc failed: {e}\n", flush=True)
            return None
        else:
            return parallel_conf

    # partition_uniform(num_layers, pipeline_size, num_chunks)
    def partition_uniform(self, num_layers, pipeline_parallel_size, num_chunks):
        assert (
            num_layers % num_chunks == 0
        ), "Layer length should be divided by the number of chunks, otherwise parameter method is recomended"

        parts = [[] for _ in range(pipeline_parallel_size)]
        partition_items = num_layers // num_chunks
        for idx in range(num_chunks):
            base_idx = idx * partition_items
            chunk_size = partition_items // pipeline_parallel_size
            left = pipeline_parallel_size - partition_items % pipeline_parallel_size
            if chunk_size == 0:
                raise ValueError("Some nodes in Pipeline have no requests")

            for p in range(pipeline_parallel_size):
                st = base_idx
                # 由于 (p >= left), head必然会被分配到后面的pp rank上，因此峰值只需要考虑 pp0 + embedding
                base_idx += chunk_size + (p >= left)
                parts[p].append((st, base_idx))

        indexes = []
        indexes_split = []
        max_layers = 0
        for pp_rank, _parts in enumerate(parts):
            indexes_split.append([])
            for s, e in _parts:
                indexes.extend(list(range(s, e)))
                indexes_split[pp_rank].extend(list(range(s, e)))
            if len(indexes_split[pp_rank]) > max_layers:
                max_layers = len(indexes_split[pp_rank])

        assert num_chunks == 1, "not support num_chunks > 1!"
        assert len(indexes) == len(set(indexes)), indexes  # should have no duplicates
        assert set(indexes) == set(list(range(num_layers))), (
            indexes,
            num_layers,
        )  # should have the same indexes as expected
        # 我们需要知道 pp rank0 被分到了多少个 layer
        # 以及被分到最多layer的pprank被分到了多少层.
        return len(indexes_split[0]), max_layers

    def run_flexible_worldsize_loop(self):
        max_node_num = self.max_world_size // 8
        min_node_num = self.min_world_size // 8
        for node_num in range(max_node_num, min_node_num - 1, -1):
            # TODO: 增加静态显存check,如果低于最低卡数要求直接break
            # print(f"node_num : {node_num}")
            world_size = node_num * 8
            solutions_list = self.run_loop(world_size)

        if self.min_cost_solution is not None:
            solutions_list = sorted(solutions_list, key=lambda solu: solu.tgs, reverse=True)
            print("--------------------- END -----------------------", flush=True)
            print("Max TGS:", self.min_comm_cost * -1)
            for i, solu in enumerate(solutions_list):
                if i > 5:
                    break
                print(f"Top{i} Solution:", solu, flush=True)

            print("--------------------- MSP best solution -----------------------", flush=True)
            if self.msp_min_solu is not None:
                print(f"self.msp_min_solu : {self.msp_min_solu}")
            print("--------------------- FSP best solution -----------------------", flush=True)
            if self.fsp_min_solu is not None:
                print(f"self.fsp_min_solu : {self.fsp_min_solu}")
            print("--------------------- ISP best solution -----------------------", flush=True)
            if self.isp_min_solu is not None:
                print(f"self.isp_min_solu : {self.isp_min_solu}")

            final_res = {
                "algo_type": self.min_cost_solution.algo_type,
                "seq_len": self.min_cost_solution.seq_len,
                "micro_num": self.min_cost_solution.micro_num,
                "micro_bsz": self.min_cost_solution.micro_bsz,
                "pp_size": self.min_cost_solution.pp,
                "tp_size": self.min_cost_solution.sp,
                "wp_size": self.min_cost_solution.wp_size,
                "zp_size": self.min_cost_solution.zp_size,
                "activation_ckpt": True if self.min_cost_solution.activation_ckpt else False,
            }
            print(final_res)
        else:
            print("No solution found")

    def cal_cost(
        self,
        pp,
        sp,
        wp,
        zp,
        micro_bsz,
        micro_num,
        algo_type,
        world_size,
        activation_ckpt,
        pp_num_layers=None,
        max_pp_num_layers=None,
    ) -> LinsSolutionNoZ3:
        if pp_num_layers is None or max_pp_num_layers is None:
            pp_num_layers, max_pp_num_layers = self.partition_uniform(
                num_layers=self._l, pipeline_parallel_size=pp, num_chunks=1
            )

        # 反碎片化惩罚
        if algo_type in [AlgoType.MSP, AlgoType.FSP]:
            if sp * zp * wp * pp < (self.model_size / 1.5):
                if self.debug:
                    print(f"NO solu: skip sp*zp*wp*pp< 4 solu!\n", flush=True)
                return None
        else:
            if zp * wp * pp < (self.model_size / 1.5):
                if self.debug:
                    print(f"NO solu: skip zp*wp*pp< 4 solu!\n", flush=True)
                return None

        # build device mesh
        parallel_conf = self.build_parallel_config(
            algo_type, world_size, pp, sp, wp=wp, zp=zp
        )  # 建立device mesh, 在build gpc的时候会筛掉一些不合理的解
        if parallel_conf is None:
            # A[pp_i][sp_i][wp_i][zp_i] = _100GB
            # C[pp_i][sp_i][wp_i][zp_i] = 0
            return None

        now_global_bsz = micro_bsz * micro_num * self.seq_len * gpc.get_world_size(ParallelMode.DATA)

        dp = gpc.get_world_size(ParallelMode.DATA)
        one_layer_elem = cal_block_p_elem(self._h, multiple_of=self.multiple_of, mlp_ratio=self.mlp_ratio)
        pp_blocks_elem = pp_num_layers * one_layer_elem
        embedding_dp_shared_range = 1 if dp <= 1 else 2
        head_num = 1 if pp > 1 else 2
        embedding_elem = self.vocab_size * self._h

        if algo_type in [AlgoType.MSP, AlgoType.FSP]:
            embedding_elem_parallel = head_num * embedding_elem / wp / sp
            block_elem_parallel = pp_blocks_elem / wp / sp
            total_p_element = block_elem_parallel + embedding_elem_parallel
            total_os_element = total_p_element / zp
            os_mm_cost = self.dtype_size * self.fp32_ratio * 3 * total_os_element  # zp显存消耗
            p_g_mm_cost = 2 * self.dtype_size * total_p_element  # wp显存消耗
        else:
            embedding_elem_parallel = head_num * embedding_elem / sp
            block_elem_parallel = pp_blocks_elem / wp
            total_p_element = block_elem_parallel + embedding_elem_parallel
            total_os_element = (
                block_elem_parallel / zp + embedding_elem_parallel / embedding_dp_shared_range
            )  # embeding不会被zp切
            os_mm_cost = self.dtype_size * self.fp32_ratio * 3 * total_os_element  # zp显存消耗
            p_g_mm_cost = 2 * self.dtype_size * total_p_element  # wp显存消耗

        zp_comm_cost, wdp_comm_cost = self.comm_dp_cost(
            algo=algo_type,
            pp_blocks_elem=block_elem_parallel,
            embedding_elem=embedding_elem_parallel,
            zp=zp,
        )  # 计算dp相关的通信开销

        # zp_comm_cost=0
        if self.overlap_wdp:
            wdp_comm_cost = 0

        blocks_activation = get_block_threshold(
            algo=algo_type,
            micro_batch_size=micro_bsz,
            layer_num=self._l,  # 显存阈值根据pp0来计算
            sp_size=sp,
            activation_ckpt=activation_ckpt,
            hidden_dim=self._h,
            sequence_length=self.seq_len,  # 这里一定要传入没切过的seqlen
            use_fa=self.use_fa,
            head_num=self._a,
            dtype_size=self.dtype_size // 2,  # dtype_size要除以2，因为激活值计算公式是默认按照fp16类型来的
        )  # isp激活的话，不需要除以wp，因为需要allgather

        if algo_type == AlgoType.ISP:
            isp_mem_pool = get_memory_pool_mm(self.mlp_ratio, self._h, self.dtype_size)
        else:
            isp_mem_pool = 0

        pp_p2p_buffer = get_p2p_buffer_size(self.dtype_size, self.seq_len, sp, micro_bsz, self._h) if pp > 1 else 0

        # 下面这些激活的计算不受到重计算的影响
        norm_activation = get_norm_output_mm(micro_bsz, self.seq_len, self._h, sp=sp, dtype_size=self.dtype_size)
        head_input_activation = get_head_input_mm(
            micro_bsz,
            self.seq_len,
            self._h,
            dtype_size=self.dtype_size,
            tp_size=sp,
            algo=algo_type,
        )
        head_output_activation = get_head_output_mm(
            micro_bsz, self.seq_len, self.vocab_size, dtype_size=self.dtype_size
        )
        rotary_emb_sincos_cache_mm = get_rotary_emb_sincos_cache_mm(
            seq_len=self.seq_len,
            pp_size=pp,
            hidden_dim=self._h,
            head_nums=self._a,
            layer_nums=self._l,
            dtype_size=self.dtype_size,
        )
        # 对于pp0,占用的激活仍然是 layer_num 份
        block_output_activation = (
            pp_num_layers
            * pp
            * get_block_output_mm(micro_bsz, self.seq_len, self._h, sp=sp, dtype_size=self.dtype_size)
        ) * activation_ckpt  # 只有开启重计算才需要额外加上这部分block激活的输出
        backward_mem_peak = get_backward_mem_peak(
            seq_len=self.seq_len,
            micro_bsz=micro_bsz,
            dtype_size=self.dtype_size,
            vocab_size=self.vocab_size,
            tp_size=sp,
            hidden_size=self._h,
        )
        activation = (
            blocks_activation
            + norm_activation
            + head_input_activation
            + head_output_activation
            + block_output_activation
            + backward_mem_peak
        )

        # 总显存开销
        mem_cost1 = (
            p_g_mm_cost + os_mm_cost + activation + isp_mem_pool + rotary_emb_sincos_cache_mm + pp_p2p_buffer
        )  # fwd_bwd显存峰值(需要加上Grad吗？)
        mem_cost2 = p_g_mm_cost + os_mm_cost / 3 * 5  # adamw的显存峰值
        mem_cost = max(mem_cost1, mem_cost2)
        if mem_cost > self.mem_threshold:
            # A[pp_i][sp_i][wp_i][zp_i] = _100GB
            # C[pp_i][sp_i][wp_i][zp_i] = 0
            if self.debug:
                print(
                    f"NO solu: mem_cost: {mem_cost/1024**3:.2f} GB > mem_threshold: {self.mem_threshold/1024**3:.2f} GB ---- p_g_mm_cost: {p_g_mm_cost/1024**3:.2f} GB, os_mm_cost: {os_mm_cost/1024**3:.2f} GB, activation: {activation/1024**3:.2f} GB\n",
                    flush=True,
                )
            return None
        # else:
        #     A[pp_i][sp_i][wp_i][zp_i] = mem_cost

        try:
            (wp_comm_cost, sp_comm_cost, comp_wp, comp_attn,) = TransformerOverlapOneLayer(
                micro_bsz=micro_bsz,
                sp_size=sp,
                pp_size=pp,
                world_size=world_size,
                ckpt=activation_ckpt,
                seq_len=self.seq_len,  # 这里需要传原始的seqlen,因为这个类里面还会切sp
                vocab_size=self.vocab_size,
                dtype_size=self.dtype_size,
                hidden_dim=self._h,
                num_head=self._a,
                mlp_ratio=self.mlp_ratio,
                multiple_of=self.multiple_of,
            )._get_overlap(algo_type)
        except KeyError as e:
            print(f"not found FA key: {e}", flush=True)
            return None

        # if wp > 1 and sp > 1:
        #     if self.model_size < 30 and self.seq_len < 16 * 1024:
        #         # 我们对overlap进行惩罚，优先级：切os->切梯度->切参数
        #         penalty_coefficient = wp / 100
        #         overlap_latency = (1 + penalty_coefficient) * max(comp_wp, wp_comm_cost)
        #     else:
        #         overlap_latency = max(comp_wp, wp_comm_cost)
        # else:
        #      overlap_latency =  max(comp_wp, wp_comm_cost)

        if wp > 1:
            overlap_latency = min(comp_wp, wp_comm_cost) * self._wp_penalty_coefficient + max(comp_wp, wp_comm_cost)
        else:
            overlap_latency = comp_wp

        def overlaped_fwd_bwd_cost():
            return overlap_latency + sp_comm_cost + comp_attn

        if pp == 1:
            fwd_bwd_cost = self._l * overlaped_fwd_bwd_cost()
            grad_acc = micro_num
            all_fwd_bwd_cost = grad_acc * fwd_bwd_cost  # 算上梯度累积的fwdbwd开销
            pp_comm_cost = 0
        else:
            # 注意这里要使用 max_pp_num_layers 来计算pp的延迟，而不是pp0的 num layer
            fwd_bwd_cost = max_pp_num_layers * overlaped_fwd_bwd_cost()  # 1个pp micro step的fwd_bwd开销
            all_fwd_bwd_cost = micro_num * fwd_bwd_cost  # pp的idea开销(不含bubble)
            pp_p2p_cost = self.pp_comm_overhead(
                pp_size=pp,
                sp_size=sp,
                micro_bsz=micro_bsz,
                micro_num=micro_num,
            )  # pp的p2p延迟
            pp_bubble_cost = (pp - 1) * fwd_bwd_cost  # pp的bubble开销
            pp_comm_cost = pp_p2p_cost + pp_bubble_cost  # pp总的额外开销

        total_latency = all_fwd_bwd_cost + pp_comm_cost + wdp_comm_cost + zp_comm_cost  # fwd_bwd_cost 乘上梯度累加

        # 计算tgs,为了方便取max这里乘了一个-1
        tgs = (-1 * now_global_bsz) / (world_size * total_latency)

        solu = LinsSolutionNoZ3(
            pp=pp,
            sp=sp,
            wp=wp,
            zp=zp,
            seq_len=self.seq_len,
            micro_bsz=micro_bsz,
            micro_num=micro_num,
            algo_type=algo_type,
            pp_comm_cost=pp_comm_cost,
            activation=activation,
            zp_comm_cost=zp_comm_cost,
            wp_comm_cost=wp_comm_cost,
            sp_comm_cost=sp_comm_cost,
            os_mm_cost=os_mm_cost,
            p_g_mm_cost=p_g_mm_cost,
            fwd_bwd_cost=fwd_bwd_cost,
            mem_cost=mem_cost,
            comp_wp=comp_wp,
            comp_attn=comp_attn,
            world_size=world_size,
            activation_ckpt=activation_ckpt,
            tgs=-1 * tgs,
            mem_pool_mm=isp_mem_pool,
            norm_activation=norm_activation,
            head_input_activation=head_input_activation,
            head_output_activation=head_output_activation,
            block_output_activation=block_output_activation,
            wdp_comm_cost=wdp_comm_cost,
            all_fwd_bwd_cost=all_fwd_bwd_cost,
            g_bsz=now_global_bsz,
            pp_p2p_buffer=pp_p2p_buffer,
            rotary_emb_sincos_cache_mm=rotary_emb_sincos_cache_mm,
            modelsize=self._param_elements / 10**9,
            backward_mem_peak=backward_mem_peak,
            blocks_activation=blocks_activation,
            overlap_latency=overlap_latency,
            total_latency=total_latency,
        )

        gpc.destroy()  # 销毁device mesh
        return solu

    def run_loop(self, world_size):
        pp_search_range = PPIter(world_size, self._l)
        sp_search_range = SPIter(world_size, self._a)
        wp_search_ranges = SPIter(world_size, world_size)
        # zp_search_ranges_max = SPIter(world_size, world_size)
        solutions_list = []

        # A = [
        #     [[[0 for _ in range(world_size)] for _ in wp_search_ranges] for _ in sp_search_range]
        #     for _ in pp_search_range
        # ]

        # C = copy.deepcopy(A)

        for pp_i, pp in enumerate(pp_search_range):
            pp_num_layers, max_pp_num_layers = self.partition_uniform(
                num_layers=self._l, pipeline_parallel_size=pp, num_chunks=1
            )
            print(f"pp_num_layers: {pp_num_layers}, pp_num_layers:{max_pp_num_layers}, pp:{pp}", flush=True)
            for sp_i, sp in enumerate(sp_search_range):
                if not self.use_fixed_micro_bsz:
                    if self._use_strict_bsz:
                        bs_bns = self.get_bsz_strict(world_size, pp, sp, self.seq_len)
                    else:
                        bs_bns = self.get_bsz_approximate(world_size, pp, sp, self.seq_len)
                    if bs_bns is None or len(bs_bns) == 0:
                        if self.debug:
                            print(
                                f"NO solu: pp:{pp} , sp:{sp} can't find micro_bsz/micro_num for"
                                f"world_size:{world_size}, seq_len:{self.seq_len}, global bsz range: [{self.global_bsz_min}-{self.global_bsz_max}]!",
                                flush=True,
                            )
                        continue
                else:
                    bs_bns = [(self.fixed_micro_bsz, self.fixed_micro_num)]

                for micro_bsz, micro_num in bs_bns:
                    for algo_type in self._algo_list:
                        for activation_ckpt in [0, 1]:
                            for wp_i, wp in enumerate(wp_search_ranges):
                                if algo_type in [AlgoType.MSP, AlgoType.FSP]:
                                    if wp > 1:
                                        if self.debug:
                                            print("NO solu: msp, fsp not support wp>1 !", flush=True)
                                        continue  # msp, fsp禁掉fsdp，我们目前还不支持
                                    # zp的搜索空间是被wp限制的，同时他不是按照8的倍数变化的，是,1,2,3, ...这样递增的
                                    zp_search_range = world_size // pp // sp // wp  # 这里的sp对于msp和fsp来说是tp
                                else:
                                    zp_search_range = (
                                        world_size // pp // wp
                                    )  # internlm实现的zp和deepspeed不一样，zp是在切wp的基础上再切的

                                try:
                                    assert self._h % sp == 0, f"embed_dim:{self._h} must be divisible by sp: {sp}"
                                    assert self._a % sp == 0, f"num_heads: {self._a} must be divisible by sp: {sp}"
                                    assert self._a >= sp, f"num_heads: {self._a} must bigger then sp: {sp}"
                                except AssertionError as e:
                                    if self.debug:
                                        print(f"NO solu: head assert {e}", flush=True)
                                    continue

                                for zp_i, zp in enumerate(range(1, zp_search_range + 1)):
                                    if self.debug:
                                        print(
                                            f"------------------- Begin: world_size: {world_size}, pp:{pp}, sp:{sp}, micro_bsz:{micro_bsz}, micro_num:{micro_num}, algo_type:{algo_type}, wp:{wp}, zp:{zp} ckpt:{activation_ckpt} -------------------",
                                            flush=True,
                                        )
                                    solu = self.cal_cost(
                                        pp=pp,
                                        sp=sp,
                                        wp=wp,
                                        zp=zp,
                                        micro_bsz=micro_bsz,
                                        micro_num=micro_num,
                                        algo_type=algo_type,
                                        world_size=world_size,
                                        activation_ckpt=activation_ckpt,
                                        pp_num_layers=pp_num_layers,
                                        max_pp_num_layers=max_pp_num_layers,
                                    )
                                    if solu is None:
                                        continue
                                    cost = solu.tgs
                                    solutions_list.append(solu)
                                    if cost < self.min_comm_cost:
                                        self.min_comm_cost = cost
                                        self.min_cost_solution = solu

                                    print(f"solu: {solu}", flush=True)

                                    if algo_type == AlgoType.MSP:
                                        if cost < self.msp_min_cost:
                                            self.msp_min_cost = cost
                                            self.msp_min_solu = solu
                                    elif algo_type == AlgoType.FSP:
                                        if cost < self.fsp_min_cost:
                                            self.fsp_min_cost = cost
                                            self.fsp_min_solu = solu
                                    elif algo_type == AlgoType.ISP:
                                        if cost < self.isp_min_cost:
                                            self.isp_min_cost = cost
                                            self.isp_min_solu = solu
        return solutions_list
