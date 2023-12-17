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


class LinsSolution:
    def __init__(
        self,
        search_ranges,
        num_strategies,
        pp,
        sp,
        micro_bsz,
        micro_num,
        mem_cost,
        algo_type,
        solution,
        C,
        A,
        pp_cost,
        activation,
        comm_cost,
    ):
        self.C = C
        self.A = A
        self.pp = pp
        self.sp = sp
        self.micro_bsz = micro_bsz
        self.micro_num = micro_num
        self.algo_type = algo_type
        self.pp_cost = pp_cost
        self.activation = activation
        self.search_ranges = search_ranges
        if solution is not None:
            # TODO: wp_size 和 zp_size 现在还是(8, 16, 32, ..)这样的搜索空间，需要补充 2,4,6
            self.wp_size = int(np.array(list(range(num_strategies)))[np.array(solution[0], dtype="bool")])
            self.zp_size = int(np.array(list(range(num_strategies)))[np.array(solution[2], dtype="bool")])
            self.zp_comm_cost = self.C[2][self.zp_size]
            self.wp_comm_cost = self.C[0][self.wp_size]
            self.zp_mm_cost = self.A[2][self.zp_size]
            self.wp_mm_cost = self.A[0][self.wp_size]
            self.mem_cost = mem_cost
            self.total_mm_cost = self.mem_cost + self.activation
            self.comm_cost = comm_cost
        else:
            self.comm_cost = -1
            self.total_mm_cost = -1
            self.wp_size = -1
            self.zp_size = -1
            self.zp_comm_cost = -1
            self.wp_comm_cost = -1
            self.zp_mm_cost = -1
            self.wp_mm_cost = -1
            self.mem_cost = -1

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return (
            f" pp: {self.pp}"
            f" sp: {self.sp}"
            f" micro_bsz: {self.micro_bsz}"
            f" micro_num: {self.micro_num}"
            f" algo_type: {self.algo_type}, wp_size: {self.search_ranges[self.wp_size]}, zp_size: {self.search_ranges[self.zp_size]}"
            f" total comm_cost: {self. comm_cost*10**3/10**4:.2f} ms, pp_comm_cost: {self.pp_cost*10**3/10**4:.2f} ms, zp_comm_cost: {self.zp_comm_cost*10**3/10**4:.2f} ms, wp_comm_cost: {self.wp_comm_cost*10**3/10**4:.2f} ms"
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


class Simulator:
    def __init__(self, num_strategies, C, A) -> None:
        self._memory_threshold = _79GB
        self._num_strategies = num_strategies

        self._X = self.get_num_strategies()  # 解空间
        self._C = C  # 通信开销
        self._A = A  # 显存开销

    def set_strategy_constraint_strict(self):
        for i in range(len(self._A)):  # xyt: 对于P、G、OS只可能有一种切分策略
            self._solver.add(z3.Sum([z3.If(self._X[i][j], 1, 0) for j in range(self._num_strategies)]) == 1)
        for j in range(1, self._num_strategies):  # xyt:感觉这里是想说明P和G的切分是绑定在一起的
            self._solver.add(self._X[0][j] == self._X[1][j])
        for j in range(1, self._num_strategies):  # 这个约束条件保证os的切分不能小于G的切分
            self._solver.add(z3.Implies(self._X[1][j], z3.Not(self._X[2][j - 1])))

    def set_memory_constraint_strict(self):
        # TODO: 可能有两个限制，一个是forward+backward；一个是step，这两个阶段都需满足显存不超过80GB
        self.total_memorycost_expr = z3.Sum(
            [
                self._A[i][j] * z3.If(self._X[i][j], 1, 0)
                for i in range(len(self._A))
                for j in range(self._num_strategies)
            ]
        )
        self._solver.add(self.total_memorycost_expr < self._memory_threshold)

    def build_constraint(self):
        self.set_strategy_constraint_strict()
        self.set_memory_constraint_strict()

    def build_optimize_object(self):
        self._total_comm_cost = z3.Int("total_comm_cost")  # TODO： 实数->整数,不要小数
        self._total_mem_cost = z3.Int("total_mem_cost")

        self._communication_cost_expr = z3.Sum(
            [self._C[i][j] * z3.If(self._X[i][j], 1, 0) for i in range(3) for j in range(self._num_strategies)]
        )
        self._solver.add(self._total_comm_cost == self._communication_cost_expr)
        self._solver.add(self._total_mem_cost == self.total_memorycost_expr)

    def get_num_strategies(self):
        return [[z3.Bool(f"X_{i}_{j}") for j in range(self._num_strategies)] for i in range(3)]

    def run(self):
        z3.set_option(precision=30)
        self._solver = z3.Solver()
        self.build_constraint()
        self.build_optimize_object()
        # self._solver.push()
        min_comm_cost, solution, re_mem_cost = None, None, None
        while self._solver.check() == z3.sat:
            model = self._solver.model()
            # current_cost = model[self._total_comm_cost].as_long()
            current_cost_value = int(str(model[self._total_comm_cost]))  # .as_decimal(10)
            current_mem_cost_value = int(str(model[self._total_mem_cost]))  # .as_decimal(10)

            if min_comm_cost is None or current_cost_value < min_comm_cost:
                min_comm_cost = current_cost_value
                solution = [[model.evaluate(self._X[i][j]) for j in range(self._num_strategies)] for i in range(3)]
                re_mem_cost = current_mem_cost_value
            # solver.add(self._total_comm_cost < current_cost)  # Add constraint to find lower cost
            self._solver.add(self._total_comm_cost < current_cost_value)

        return min_comm_cost, solution, re_mem_cost


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

    def mem_cost(self, i: int, j: int, sp_size: int, model_p_element: int, algo_type: AlgoType):
        """_summary_

        Args:
            i (int): i = 0 代表 wp, j=2 代表 zp
            j (int): _description_
            sp_size (int): _description_
            model_p_element (int): _description_
            algo_type (AlgoType): _description_

        Returns:
            memory cost: 显存开销, 单位为B
        """
        if i == 0:
            if j == 0:
                # 不切P(无wp)
                cost = 2 * self.dtype_size * model_p_element  # (P + G)
                if algo_type in [AlgoType.MSP, AlgoType.FSP]:
                    return cost / sp_size
                else:
                    return cost
            else:
                # 对P切j*8份(有wp)
                cost = 2 * self.dtype_size * model_p_element / (j * 8)
                if algo_type in [AlgoType.MSP, AlgoType.FSP]:
                    return cost / sp_size
                else:
                    return cost
        elif i == 2:
            if j == 0:
                # 不切OS
                return self.dtype_size * self.fp32_ratio * 3 * model_p_element
            else:
                # 对OS切j*8份
                return self.dtype_size * self.fp32_ratio * 3 * model_p_element / (j * 8)
        else:
            return 0  # no cost

    def get_cost(
        self,
        search_ranges,
        num_strategies,
        algo_type,
        world_size,
        micro_bsz,
        micro_num,
        activation_ckpt,
        sp,
        pp,
        model_p_element,
    ):
        SEARCH_DIMENSION = len(["pp", "sp", "wp", "zp"])
        # 2,4,6,8,16,32
        A = [
            # [0 for _ in range(num_strategies)], # pp
            # [0 for _ in range(num_strategies)], # sp
            [0 for _ in range(num_strategies)],  # wp
            [0 for _ in range(num_strategies)],  # gp
            [0 for _ in range(num_strategies)],  # zp
        ]
        C = copy.deepcopy(A)

        pp_model_p_element = model_p_element // pp
        grad_acc = micro_num if pp == 1 else 1
        memory_threshold, activation = TransformerMemory(
            self.dtype_size,
            pp,
            sp,
            micro_bsz,
            self.seq_len,
            self.model_size,
            activation_ckpt,
            self.use_fa,
        ).get_memory_threshold(algo_type)

        for i in range(len(A)):
            for j in range(len(A[0])):
                shared_scale = search_ranges[j]
                # print(f"i: {i}, j: {j}, shared_scale:{shared_scale}", flush=True)
                # lins_scale = j * 8 if j != 0 else 1

                if j == 0:
                    A[i][j] = _100GB  # 不可行解
                    C[i][j] = 0
                    continue

                # 显存开销
                A[i][j] = self.mem_cost(i, shared_scale, sp, model_p_element, algo_type) + activation

                # 通信开销
                if i == 0:
                    parallel_conf = self.build_parallel_config(algo_type, world_size, pp, sp, wp=shared_scale, zp=1)
                    if parallel_conf is None:
                        A[i][j] = _100GB
                        C[i][j] = 0
                    else:
                        C[i][j] = grad_acc * TransformerOverlap(  # 梯度累加
                            micro_bsz=micro_bsz,
                            seq_len=self.seq_len,
                            vocab_size=self.vocab_size,
                            dtype_size=self.dtype_size,
                            model_size=self.model_size,
                            sp_size=sp,
                            pp_size=pp,
                            world_size=self.world_size,
                            ckpt=activation_ckpt,
                            model_para=pp_model_p_element,  # TODO: 这里除了 PP 需要 check 正确性
                            num_layers=self.num_layer,
                        )._get_overlap(shared_scale, algo_type)
                elif i == 2:
                    parallel_conf = self.build_parallel_config(algo_type, world_size, pp, sp, wp=1, zp=shared_scale)
                    if parallel_conf is None:
                        A[i][j] = _100GB
                        C[i][j] = 0
                    else:
                        C[i][j] = self.comm_os_cost(model_p_element)
                        print(f"C[{i}][{j}]: {C[i][j]}", flush=True)
                else:
                    C[i][j] = 0  # 暂时不用

                gpc.destroy()  # 销毁device mesh

        return A, C, activation

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
        except AssertionError as e:
            # print(f"AssertionError: algo_type:{algo_type}, world_size:{world_size}, pp:{pp}, sp:{sp}, wp:{wp}, zp:{zp}")
            return None
        else:
            return parallel_conf

    def run_loop(self):
        min_comm_cost = float("inf")
        min_cost_solution = None
        possible_solution = []
        for pp in PPIter(self.world_size, self._l):
            for sp in SPIter(self.world_size // pp, self._a):
                bs_bns = self.get_bsz(pp, sp, self.seq_len)
                # the layer number should be updated
                self.num_layer = self._l // pp
                for micro_bsz, micro_num in bs_bns:
                    pp_model_p_element = self._param_elements // pp
                    for algo_type in self._algo_list:
                        for activation_ckpt in [
                            0,
                        ]:  # the value should be {0, 1}
                            pp_comm_cost = self.pp_comm_overhead(pp, sp, self.seq_len, micro_bsz, micro_num, self._h)

                            # 这里不限制head数量，只是为了获得 2,4,6,8,16这样的解空间
                            search_ranges = [0] + list(SPIter(self.world_size, self.world_size))
                            num_strategies = len(search_ranges)  # + 1 表示从下标1开始搜素

                            A, C, activation = self.get_cost(
                                search_ranges,
                                num_strategies,
                                algo_type,
                                self.world_size,
                                micro_bsz=micro_bsz,
                                micro_num=micro_num,
                                activation_ckpt=activation_ckpt,
                                sp=sp,
                                pp=pp,
                                model_p_element=self._param_elements,
                            )

                            simulator = Simulator(num_strategies, C=C, A=A)
                            comm_cost, solution, mem_cost = simulator.run()
                            if comm_cost is not None:
                                comm_cost += pp_comm_cost

                            solu = LinsSolution(
                                search_ranges,
                                num_strategies,
                                pp,
                                sp,
                                micro_bsz,
                                micro_num,
                                mem_cost,
                                algo_type,
                                solution,
                                C,
                                A,
                                pp_comm_cost,
                                activation,
                                comm_cost,
                            )

                            if comm_cost is not None:
                                if comm_cost < min_comm_cost:
                                    min_comm_cost = comm_cost
                                    min_cost_solution = solu
                                possible_solution.append(solu)
                                print(f"solu: {solu}", flush=True)
                            else:
                                print(f"no-solu: {solu}", flush=True)

        if min_cost_solution is not None:
            print("Minimum Communication Cost:", min_comm_cost)
            print("Solution:", min_cost_solution, flush=True)
        else:
            print("No solution found")