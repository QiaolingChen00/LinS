import copy
import math

from simulator.ab_cost_model import allreduce, broadcast, p2p
from simulator.context import ParallelMode, check_and_modify_parallel_config
from simulator.context import global_context as gpc
from simulator.mem import get_memory_threshold
from simulator.overlap import TransformerOverlapOneLayer

# from comm import TransformerCommunication
# from utils.utils import _get_model_config
from utils.common import _79GB, _100GB, GB, AlgoType, get_model_config
from utils.config import Config


class LinsSolutionNoZ3:
    def __init__(
        self,
        pp,
        sp,
        wp,
        zp,
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
    ):
        self.pp = pp
        self.sp = sp
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

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return (
            f" world_size: {self.world_size}"
            f" pp: {self.pp}"
            f" sp: {self.sp}"
            f" activation ckpt: {self.activation_ckpt}"
            f" micro_bsz: {self.micro_bsz}"
            f" micro_num: {self.micro_num}"
            f" algo_type: {self.algo_type}, wp_size: {self.wp_size}, zp_size: {self.zp_size}"
            f" total fwd_bwd_cost: {self. fwd_bwd_cost*10**3/10**4:.2f} ms, pp_comm_cost: {self.pp_comm_cost*10**3/10**4:.2f} ms, zp_comm_cost: {self.zp_comm_cost*10**3/10**4:.2f} ms, wp_comm_cost: {self.wp_comm_cost*10**3/10**4:.2f} ms, sp_comm_cost: {self.sp_comm_cost*10**3/10**4:.2f} ms"
            f" comp_wp: {self.comp_wp*10**3/10**4:.2f} ms, comp_attn: {self.comp_attn*10**3/10**4:.2f} ms"
            f" total mem_cost: {self.total_mm_cost /GB:.2f} GB, activation: {self.activation/GB:.2f} GB, os_mm_cost: {self.os_mm_cost/GB:.2f} GB, p_g_mm_cost: {self.p_g_mm_cost/GB:.2f} GB"
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
        world_size,
        global_bsz,
        global_bsz_min,
        global_bsz_max,
        max_world_size,
        min_world_size,
        seq_len,
        debug,
        config,
    ) -> None:
        self.world_size = world_size
        self.global_bsz = global_bsz  # 4
        self.global_bsz_min = global_bsz_min  # 4
        self.global_bsz_max = global_bsz_max  # 5
        self.max_world_size = max_world_size
        self.min_world_size = min_world_size
        self.debug = debug

        self.seq_len = seq_len
        self.dtype_size = config.dtype_size
        self.model_size = config.model_size
        self.vocab_size = config.vocab_size
        self.use_fa = config.use_fa
        self.fp32_ratio = max(1, 4 // self.dtype_size)
        self._param_elements = self.model_size * 10**9
        self._param_size_in_byte = self.model_size * self.dtype_size * 10**9
        self._h, self._a, self._l, self.mlp_ratio, self.multiple_of = get_model_config(self.model_size)
        self._algo_list = [AlgoType.ISP, AlgoType.MSP, AlgoType.FSP]

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

        num_tokens = self.global_bsz
        dp_world_size = world_size // pp_size // sp_size
        bsz = num_tokens // dp_world_size // seq_len

        micro_bsz_num = []
        for micro_bsz in range(1, int(bsz**0.5) + 1):
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

    def pp_comm_overhead(self, pp_size, sp_size, seq_len, micro_bsz, micro_num):
        """计算pp中p2p通信的延迟"""
        if pp_size == 1:
            return 0

        p2p_buffer_size = self.dtype_size * (seq_len // sp_size) * micro_bsz * self._h

        warmup_p2p_num = min(pp_size, micro_num)
        one_f_one_b_p2p_num = micro_num - 1
        cooldown_p2p_num = min(pp_size, micro_num)

        p2p_latency = (warmup_p2p_num + one_f_one_b_p2p_num + cooldown_p2p_num) * p2p(
            p2p_buffer_size, ParallelMode.PIPELINE
        )
        return p2p_latency

    def comm_dp_cost(self, algo, pp_model_element, sp_size) -> float:
        """切分OS引入的参数同步的通信开销"""

        # zero引入的参数同步开销, 这里传入的是一个dp rank的通信量
        shared_nums = gpc.get_world_size(ParallelMode.ZERO1)
        buffer_size = self.dtype_size * pp_model_element // shared_nums
        zp_latency = shared_nums * broadcast(buffer_size, ParallelMode.ZERO1)

        # wdp引入的通信开销, 这里传入的是一个dp rank视角下的通信量
        # msp和fsp的参数被tp切了，所需需要除以sp_size
        if algo == AlgoType.MSP:
            wdp_latency = allreduce(pp_model_element // sp_size, ParallelMode.WEIGHT_DATA)
        elif algo == AlgoType.FSP:
            wdp_latency = allreduce(pp_model_element // sp_size, ParallelMode.WEIGHT_DATA)
        elif algo == AlgoType.ISP:
            wdp_latency = allreduce(pp_model_element, ParallelMode.WEIGHT_DATA)

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
                print(f"NO solu: build gpc assertion error: {e}", flush=True)
            return None
        else:
            return parallel_conf

    def run_flexible_worldsize_loop(self):
        max_node_num = self.max_world_size // 8
        min_node_num = self.min_world_size // 8
        for node_num in range(max_node_num, min_node_num - 1, -1):
            # TODO: 增加静态显存check,如果低于最低卡数要求直接break
            # print(f"node_num : {node_num}")
            world_size = node_num * 8
            self.run_loop(world_size)

        if self.min_cost_solution is not None:
            print("Minimum Communication Cost:", self.min_comm_cost)
            print("Solution:", self.min_cost_solution, flush=True)
            if self.msp_min_solu is not None:
                print(f"self.msp_min_solu : {self.msp_min_solu}")
            if self.fsp_min_solu is not None:
                print(f"self.fsp_min_solu : {self.fsp_min_solu}")
            if self.isp_min_solu is not None:
                print(f"self.isp_min_solu : {self.isp_min_solu}")
        else:
            print("No solution found")

    def run_loop(self, world_size):
        # pp_search_range = list(reversed(list(PPIter(self.world_size, self._l))))
        # sp_search_range = list(reversed(list(SPIter(self.world_size, self._a))))
        # wp_zp_search_ranges = list(reversed(list(SPIter(self.world_size, self.world_size))))

        pp_search_range = PPIter(world_size, self._l)
        sp_search_range = SPIter(world_size, self._a)
        wp_search_ranges = SPIter(world_size, world_size)
        # zp_search_ranges_max = SPIter(world_size, world_size)

        A = [
            [[[0 for _ in range(world_size)] for _ in wp_search_ranges] for _ in sp_search_range]
            for _ in pp_search_range
        ]

        C = copy.deepcopy(A)

        for pp_i, pp in enumerate(pp_search_range):
            for sp_i, sp in enumerate(sp_search_range):
                bs_bns = self.get_bsz_approximate(world_size, pp, sp, self.seq_len)
                if bs_bns is None:
                    if self.debug:
                        print("NO solu: bs_bns is None!", flush=True)
                    continue
                # the layer number should be updated
                for micro_bsz, micro_num in bs_bns:
                    for algo_type in self._algo_list:
                        for activation_ckpt in [0, 1]:
                            pp_model_element = self._param_elements // pp  # 被pp切后的模型参数大小
                            pp_num_layers = self._l // pp  # 被pp切后的layer数量
                            sp_seq_length = self.seq_len // sp  # 被sp切后的 sequence 长度

                            for wp_i, wp in enumerate(wp_search_ranges):
                                if algo_type in [AlgoType.MSP, AlgoType.FSP] and wp_i > 1:
                                    continue  # msp, fsp禁掉fsdp，我们目前还不支持

                                # zp的搜索空间是被wp限制的，同时他不是按照8的倍数变化的，是,1,2,3, ...这样递增的
                                zp_search_range = world_size // pp // wp

                                for zp_i, zp in enumerate(range(zp_search_range)):
                                    if wp > zp:  # os切分需要大于P/G (这个需要讨论下要不要加, 如果没有这个限制会有额外通信)
                                        # if self.debug:
                                        #     print(f"wp:{wp} > zp: {zp}", flush=True)
                                        continue
                                    if self.debug:
                                        print(
                                            f"------------------- Begin: world_size: {world_size}, pp:{pp}, sp:{sp}, micro_bsz:{micro_bsz}, micro_num:{micro_num}, algo_type:{algo_type}, wp:{wp}, zp:{zp} -------------------",
                                            flush=True,
                                        )

                                    # wp显存消耗
                                    if algo_type in [AlgoType.MSP, AlgoType.FSP]:
                                        p_g_mm_cost = 2 * self.dtype_size * pp_model_element / wp / sp
                                        tp_hidden_dim = self._h // sp
                                    else:
                                        p_g_mm_cost = 2 * self.dtype_size * pp_model_element / sp
                                        tp_hidden_dim = self._h

                                    # zp显存消耗
                                    os_mm_cost = self.dtype_size * self.fp32_ratio * 3 * pp_model_element / zp

                                    activation = get_memory_threshold(
                                        algo=algo_type,
                                        micro_batch_size=micro_bsz,
                                        sequence_length=sp_seq_length,
                                        hidden_dim=tp_hidden_dim,
                                        layer_num=pp_num_layers * pp,  # 显存阈值根据pp0来计算
                                        activation_ckpt=activation_ckpt,
                                        use_fa=self.use_fa,
                                        head_num=self._a,
                                        dtype_size=self.dtype_size,
                                    )

                                    # 总显存开销
                                    mem_cost = p_g_mm_cost + os_mm_cost + activation
                                    if mem_cost > _79GB:
                                        A[pp_i][sp_i][wp_i][zp_i] = _100GB
                                        C[pp_i][sp_i][wp_i][zp_i] = 0
                                        if self.debug:
                                            print(
                                                f"NO solu: mem_cost: {mem_cost/1024**3:.2f} GB > _79GB: {_79GB} ---- p_g_mm_cost: {p_g_mm_cost/1024**3:.2f} GB, os_mm_cost: {os_mm_cost/1024**3:.2f} GB, activation: {activation/1024**3:.2f} GB",
                                                flush=True,
                                            )
                                        continue
                                    else:
                                        A[pp_i][sp_i][wp_i][zp_i] = mem_cost

                                    # build device mesh
                                    parallel_conf = self.build_parallel_config(
                                        algo_type, world_size, pp, sp, wp=wp, zp=zp
                                    )  # 建立device mesh, 在build gpc的时候会筛掉一些不合理的解
                                    if parallel_conf is None:
                                        A[pp_i][sp_i][wp_i][zp_i] = _100GB
                                        C[pp_i][sp_i][wp_i][zp_i] = 0
                                        continue

                                    # 计算dp相关的通信开销
                                    zp_comm_cost, wdp_comm_cost = self.comm_dp_cost(algo_type, pp_model_element, sp)

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

                                    def overlaped_fwd_bwd_cost():
                                        return max(wp_comm_cost, comp_wp) + sp_comm_cost + comp_attn

                                    if pp == 1:
                                        fwd_bwd_cost = self._l * overlaped_fwd_bwd_cost()
                                        grad_acc = micro_num
                                        all_fwd_bwd_cost = grad_acc * fwd_bwd_cost  # 算上梯度累积的fwdbwd开销
                                        pp_comm_cost = 0
                                    else:
                                        fwd_bwd_cost = micro_num * pp_num_layers * overlaped_fwd_bwd_cost()
                                        all_fwd_bwd_cost = micro_num * fwd_bwd_cost  # pp的idea开销(不含bubble)
                                        pp_p2p_cost = self.pp_comm_overhead(
                                            pp_size=pp,
                                            sp_size=sp,
                                            micro_bsz=micro_bsz,
                                            micro_num=micro_num,
                                            seq_len=self.seq_len,
                                        )  # pp的p2p延迟
                                        pp_bubble_cost = (pp - 1) * fwd_bwd_cost  # pp的bubble开销
                                        pp_comm_cost = pp_p2p_cost + pp_bubble_cost  # pp总的额外开销

                                    C[pp_i][sp_i][wp_i][zp_i] = (
                                        all_fwd_bwd_cost + pp_comm_cost + wdp_comm_cost + zp_comm_cost
                                    )  # fwd_bwd_cost 乘上梯度累加

                                    solu = LinsSolutionNoZ3(
                                        pp=pp,
                                        sp=sp,
                                        wp=wp,
                                        zp=zp,
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
                                    )

                                    cost = C[pp_i][sp_i][wp_i][zp_i]
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

                                    gpc.destroy()  # 销毁device mesh
