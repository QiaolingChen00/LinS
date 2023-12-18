import copy
import math
import pickle
from math import log2

import numpy as np

# from z3 import *
import z3

from simulator.ab_cost_model import broadcast, p2p
from simulator.context import ParallelMode, check_and_modify_parallel_config
from simulator.context import global_context as gpc
from simulator.mem import TransformerMemory
from simulator.overlap import TransformerOverlap

# from comm import TransformerCommunication
# from utils.utils import _get_model_config
from utils.common import _79GB, _100GB, GB, AlgoType, CostType, get_model_config
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
        zp_mm_cost,
        wp_mm_cost,
        fwd_bwd_cost,
        mem_cost,
        comp_wp,
        comp_attn,
    ):
        self.pp = pp
        self.sp = sp
        self.micro_bsz = micro_bsz
        self.micro_num = micro_num
        self.algo_type = algo_type
        self.pp_comm_cost = pp_comm_cost
        self.activation = activation

        self.wp_size = wp
        self.zp_size = zp
        self.zp_comm_cost = zp_comm_cost
        self.wp_comm_cost = wp_comm_cost
        self.zp_mm_cost = zp_mm_cost
        self.wp_mm_cost = wp_mm_cost
        self.sp_comm_cost = sp_comm_cost
        self.total_mm_cost = mem_cost
        self.fwd_bwd_cost = fwd_bwd_cost
        self.comp_wp = comp_wp
        self.comp_attn = comp_attn

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return (
            f" pp: {self.pp}"
            f" sp: {self.sp}"
            f" micro_bsz: {self.micro_bsz}"
            f" micro_num: {self.micro_num}"
            f" algo_type: {self.algo_type}, wp_size: {self.wp_size}, zp_size: {self.zp_size}"
            f" total fwd_bwd_cost: {self. fwd_bwd_cost*10**3/10**4:.2f} ms, pp_comm_cost: {self.pp_comm_cost*10**3/10**4:.2f} ms, zp_comm_cost: {self.zp_comm_cost*10**3/10**4:.2f} ms, wp_comm_cost: {self.wp_comm_cost*10**3/10**4:.2f} ms, sp_comm_cost: {self.sp_comm_cost*10**3/10**4:.2f}"
            f" self.comp_wp: {self.comp_wp*10**3/10**4:.2f} ms, self.comp_attn: {self.comp_attn*10**3/10**4:.2f} ms"
            f" total mem_cost: {self.total_mm_cost /GB:.2f} GB, activation: {self.activation/GB:.2f} GB, zp_mm_cost: {self.zp_mm_cost/GB:.2f} GB, wp_mm_cost: {self.wp_mm_cost/GB:.2f} GB"
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
    def __init__(self, world_size, global_bsz, seq_len, config) -> None:
        self.world_size = world_size
        self.global_bsz = global_bsz  # 4
        self.seq_len = seq_len
        self.dtype_size = config.dtype_size
        self.model_size = config.model_size
        self.vocab_size = config.vocab_size
        self.use_fa = config.use_fa
        self.fp32_ratio = 2
        self._param_elements = self.model_size * 10**9
        self._param_size_in_byte = self.model_size * self.dtype_size * 10**9
        self._h, self._a, self._l, self.mlp_ratio, self.multiple_of = get_model_config(self.model_size)
        self._algo_list = [AlgoType.ISP, AlgoType.MSP, AlgoType.FSP]

    def get_bsz(self, pp_size, sp_size, seq_len):
        if pp_size * sp_size > self.world_size:
            return None

        num_tokens = self.global_bsz
        dp_world_size = self.world_size // pp_size // sp_size
        bsz = num_tokens // dp_world_size // seq_len

        micro_bsz_num = []
        for micro_bsz in range(1, int(bsz**0.5) + 1):
            if bsz % micro_bsz == 0:
                micro_num = bsz // micro_bsz
                if micro_num >= pp_size:  # 我们暂时不考虑 micro_num < pp_size 的情况
                    micro_bsz_num.append((micro_bsz, micro_num))
        return micro_bsz_num

    @staticmethod
    def pp_bubble_fraction(pp_size, micro_num):
        return pp_size - 1 / micro_num

    def pp_comm_overhead(self, pp_size, sp_size, seq_len, micro_bsz, micro_num, hidden_size):
        """计算pp中p2p通信的延迟"""
        if pp_size == 1:
            return 0

        buffer_size = (
            self.dtype_size * seq_len * micro_bsz * self._h // sp_size
        )  # TODO:  这里的 hidden_size 是不是有问题，是不是需要考虑切TP的情况
        warmup_p2p_num = min(pp_size, micro_num)
        one_f_one_b_p2p_num = micro_num - 1
        cooldown_p2p_num = min(pp_size, micro_num)

        p2p_latency = (warmup_p2p_num + one_f_one_b_p2p_num + cooldown_p2p_num) * p2p(
            buffer_size, ParallelMode.PIPELINE
        )
        return p2p_latency

    def comm_os_cost(self, model_p_element) -> float:
        """切分OS引入的参数同步的通信开销"""
        return broadcast(self.dtype_size * model_p_element, ParallelMode.ZERO1)

    def build_parallel_config(self, algo_type: AlgoType, world_size, pp, sp, wp, zp):
        """TODO: add wdp,
        wdp不需要出现在config里,可以自动算出来

        """

        try:
            pipeline = dict(size=pp, interleaved_overlap=True)
            tensor = dict(size=1, sp=algo_type, intern_overlap=False, memory_pool=False)
            zero1 = dict(size=zp, fsdp=False)
            weight = dict(size=wp, overlap=True, memory_pool=True)
            sequence = dict(size=sp)

            parallel_conf = Config(
                {
                    "parallel": dict(
                        zero1=zero1,
                        tensor=tensor,
                        pipeline=pipeline,
                        weight=weight,
                        sequence=sequence,
                    )
                }
            )

            gpc.load_config(parallel_conf)
            gpc.init_global_dist(0, world_size, "nccl", 1, 1)
            gpc.init_parallel_groups()  # work globally
            check_and_modify_parallel_config(parallel_conf)
        except AssertionError:
            return None
        else:
            return parallel_conf

    def run_loop(self):
        min_comm_cost = float("inf")
        min_cost_solution = None
        possible_solution = []

        # pp_search_range = list(reversed(list(PPIter(self.world_size, self._l))))
        # sp_search_range = list(reversed(list(SPIter(self.world_size, self._a))))
        # wp_zp_search_ranges = list(reversed(list(SPIter(self.world_size, self.world_size))))

        pp_search_range = PPIter(self.world_size, self._l)
        sp_search_range = SPIter(self.world_size, self._a)
        wp_zp_search_ranges = SPIter(self.world_size, self.world_size)

        A = [
            [[[0 for _ in wp_zp_search_ranges] for _ in wp_zp_search_ranges] for _ in sp_search_range]
            for _ in pp_search_range
        ]

        C = copy.deepcopy(A)

        for pp_i, pp in enumerate(pp_search_range):
            for sp_i, sp in enumerate(sp_search_range):
                bs_bns = self.get_bsz(pp, sp, self.seq_len)
                if bs_bns is None:
                    continue
                # the layer number should be updated
                for micro_bsz, micro_num in bs_bns:
                    for algo_type in self._algo_list:
                        for activation_ckpt in [
                            0,
                        ]:  # the value should be {0, 1}
                            pp_comm_cost = self.pp_comm_overhead(pp, sp, self.seq_len, micro_bsz, micro_num, self._h)
                            model_p_element = self._param_elements // pp  # 被pp切后的模型参数大小
                            pp_num_layers = self._l // pp  # 被pp切后的layer数量

                            for wp_i, wp in enumerate(wp_zp_search_ranges):
                                for zp_i, zp in enumerate(wp_zp_search_ranges):
                                    if wp > zp:
                                        continue  # os切分需要大于P/G(这个需要讨论下要不要加)
                                    grad_acc = micro_num if pp == 1 else 1
                                    _, activation = TransformerMemory(
                                        self.dtype_size,
                                        pp,
                                        sp,
                                        micro_bsz,
                                        self.seq_len,
                                        self.model_size,
                                        activation_ckpt,
                                        self.use_fa,
                                    ).get_memory_threshold(algo_type)

                                    # wp开销
                                    wp_mm_cost = 2 * self.dtype_size * model_p_element / wp  # (P + G)
                                    if algo_type in [AlgoType.MSP, AlgoType.FSP]:
                                        wp_mm_cost = wp_mm_cost / sp  # 除以 TP 的开销

                                    zp_mm_cost = self.dtype_size * self.fp32_ratio * 3 * model_p_element / zp

                                    # 显存开销
                                    mem_cost = wp_mm_cost + zp_mm_cost + activation
                                    if mem_cost > _79GB:
                                        A[pp_i][sp_i][wp_i][zp_i] = _100GB
                                        C[pp_i][sp_i][wp_i][zp_i] = 0
                                        continue
                                        # break   # 剪枝
                                    else:
                                        A[pp_i][sp_i][wp_i][zp_i] = mem_cost

                                    # 通信开销, 在build gpc的时候会筛掉一些不合理的解
                                    parallel_conf = self.build_parallel_config(
                                        algo_type, self.world_size, pp, sp, wp=wp, zp=zp
                                    )
                                    if parallel_conf is None:
                                        A[pp_i][sp_i][wp_i][zp_i] = _100GB
                                        C[pp_i][sp_i][wp_i][zp_i] = 0
                                        continue

                                    zp_comm_cost = self.comm_os_cost(model_p_element)

                                    (wp_comm_cost, sp_comm_cost, comm_wdp, comp_wp, comp_attn,) = TransformerOverlap(
                                        micro_bsz=micro_bsz,
                                        seq_len=self.seq_len,
                                        vocab_size=self.vocab_size,
                                        dtype_size=self.dtype_size,
                                        model_size=self.model_size,
                                        sp_size=sp,
                                        pp_size=pp,
                                        world_size=self.world_size,
                                        ckpt=activation_ckpt,
                                        model_para=model_p_element,  # TODO: 这里除了 PP 需要 check 正确性
                                    )._get_overlap(algo_type)

                                    fwd_bwd_cost = pp_num_layers * (
                                        max(wp_comm_cost, comp_wp) + sp_comm_cost + comp_attn
                                    )
                                    C[pp_i][sp_i][wp_i][zp_i] = (
                                        zp_comm_cost + grad_acc * fwd_bwd_cost + pp_comm_cost + comm_wdp
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
                                        zp_mm_cost=zp_mm_cost,
                                        wp_mm_cost=wp_mm_cost,
                                        fwd_bwd_cost=fwd_bwd_cost,
                                        mem_cost=mem_cost,
                                        comp_wp=comp_wp,
                                        comp_attn=comp_attn,
                                    )

                                    if C[pp_i][sp_i][wp_i][zp_i] < min_comm_cost:
                                        min_comm_cost = C[pp_i][sp_i][wp_i][zp_i]
                                        min_cost_solution = solu

                                    possible_solution.append(solu)
                                    print(f"solu: {solu}", flush=True)

                                    gpc.destroy()  # 销毁device mesh

        if min_cost_solution is not None:
            print("Minimum Communication Cost:", min_comm_cost)
            print("Solution:", min_cost_solution, flush=True)
        else:
            print("No solution found")
