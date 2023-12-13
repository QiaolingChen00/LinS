import math
import pickle
from math import log2

# from z3 import *
import z3

from simulator.ab_cost_model import get_comm_cost
from simulator.overlap import TransformerOverlap

# from comm import TransformerCommunication
# from utils.utils import _get_model_config
from utils.common import *
from utils.common import _79GB, SovlerType
from simulator.algo import ISP, MSP, FSP
from utils.common import AlgoType


def get_model_config(model_size):
    if model_size == 7:
        h = 4096
        a = 32
        l = 32
    elif model_size == 13:
        h = 5120
        a = 40
        l = 40
    elif model_size == 20:
        h = 5120
        a = 40
        l = 60
    elif model_size == 30:
        h = 6144
        a = 48
        l = 60
    else:
        h = 8192
        a = 64
        l = 80

    mlp_ratio = 8 / 3
    multiple_of = 256

    return h, a, l, mlp_ratio, multiple_of


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


class PPIter:
    def __init__(self, gpu_nums, layer_nums):
        assert layer_nums % 2 == 0
        stop = int(math.log2(min(gpu_nums, layer_nums)))
        self.num_list = [2**i for i in range(stop + 1)]

    def __iter__(self):
        return iter(self.num_list)


class Simulator:
    def __init__(self, memory_threshold, num_strategies, C, A) -> None:
        self._memory_threshold = memory_threshold
        self._num_strategies = num_strategies

        self._X = self.get_num_strategies()  # 解空间
        self._C = C  # 通信开销
        self._A = A  # 显存开销

    def set_strategy_constraint_strict(self):
        for i in range(3):  # xyt: 对于P、G、OS只可能有一种切分策略
            self._solver.add(z3.Sum([z3.If(self._X[i][j], 1, 0) for j in range(self._num_strategies)]) == 1)
        for j in range(1, self._num_strategies):  # xyt:感觉这里是想说明P和G的切分是绑定在一起的
            self._solver.add(self._X[0][j] == self._X[1][j])
        for j in range(1, self._num_strategies):  # 这个约束条件保证os的切分不能小于G的切分
            self._solver.add(z3.Implies(self._X[1][j], z3.Not(self._X[2][j - 1])))

    def set_memory_constraint_strict(self):
        # TODO: 可能有两个限制，一个是forward+backward；一个是step，这两个阶段都需满足显存不超过80GB
        self.total_memorycost_expr = z3.Sum(
            [self._A[i][j] * z3.If(self._X[i][j], 1, 0) for i in range(3) for j in range(self._num_strategies)]
        )
        self._solver.add(self.total_memorycost_expr < self._memory_threshold)

    def build_constraint(self):
        self.set_strategy_constraint_strict()
        self.set_memory_constraint_strict()

    def build_optimize_object(self):
        self._total_comm_cost = z3.Real("total_comm_cost")
        self._total_mem_cost = z3.Real("total_mem_cost")

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
        min_cost, solution, re_mem_cost = None, None, None
        while self._solver.check() == z3.sat:
            model = self._solver.model()
            # current_cost = model[self._total_comm_cost].as_long()
            current_cost_str = model[self._total_comm_cost].as_decimal(10)
            current_mem_cost_str = model[self._total_mem_cost].as_decimal(10)

            current_cost_value = float(current_cost_str.rstrip("?"))
            current_mem_cost_value = float(current_mem_cost_str.rstrip("?"))

            if min_cost is None or current_cost_value < min_cost:
                min_cost = current_cost_value
                solution = [[model.evaluate(self._X[i][j]) for j in range(self._num_strategies)] for i in range(3)]
                re_mem_cost = current_mem_cost_value
            # solver.add(self._total_comm_cost < current_cost)  # Add constraint to find lower cost
            self._solver.add(self._total_comm_cost < current_cost_value)

        return min_cost, solution, re_mem_cost


def get_cost(a):
    return 1


def get_p2p_cost(complexity):
    return complexity // 100  # IB BW: 100GB/s


class ExternalRestraint:
    def __init__(self, world_size, global_bsz, seq_len, config, cost_data) -> None:
        self.world_size = world_size
        self.global_bsz = global_bsz  # 4
        self.seq_len = seq_len
        self.dtype_size = config.dtype_size
        self.cost_data = cost_data
        self.model_size = config.model_size
        self.vocab_size = config.vocab_size
        self.fp32_ratio = 2
        self._param_elements = self.model_size * 10**9
        self._param_size_in_byte = self.model_size * self.dtype_size * 10**9
        self._h, self._a, self._l, self.mlp_ratio, self.multiple_of = get_model_config(self.model_size)

    def get_bsz(self, pp_size, sp_size, seq_len):
        num_tokens = self.global_bsz
        dp_world_size = self.world_size // (pp_size + sp_size)
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
        buffer_size = (
            self.dtype_size * seq_len * micro_bsz * self._h // sp_size
        )  # TODO:  这里的 hidden_size 是不是有问题，是不是需要考虑切TP的情况
        warmup_p2p_num = min(pp_size, micro_num)
        one_f_one_b_p2p_num = micro_num - 1
        cooldown_p2p_num = min(pp_size, micro_num)

        p2p_latency = (warmup_p2p_num + one_f_one_b_p2p_num + cooldown_p2p_num) * get_p2p_cost(buffer_size)
        return p2p_latency

    def _get_memory_threshold(self, micro_bsz, sp_size, pp_size, layer_num, seq_len):
        # 显存阈值根据pp0来计算，需要micro_num >= pp，stage_0需要保存 pp 份才成立
        activation = (self.dtype_size * micro_bsz * seq_len * (34 * self._h + 5 * self._a * seq_len)) * layer_num
        return _79GB - activation, activation

    def _comm_cost(self, i: int, j: int, sp_size: int, overlap_res, model_p_element) -> float:
        """
        Get communication cost.

        Args:
            i (int): p (i==0), g (i==1), os (i==2)
            j (int): node count

        Returns:
            float: communication cost

        commu cost = fwd + bwd + optimizer

        fwd = sp + wp
        bwd = sp + wp
        optimizer = zp

        其中 wp的通信可以overlap
        """
        if j != 0:
            comm_range = j * 8
        else:
            comm_range = 1  # no comm cost

        # 算overlap的通信开销
        # overlap_cost = overlap_res._get_overlap(comm_range, sp_size)
        overlap_cost = 0

        # 算os的通信开销
        if comm_range == 1:
            os_comm_cost = 0
        else:
            os_comm_cost = self._get_os_comm_cost(comm_range, model_p_element)

        # 总的通信开销
        comm_cost = os_comm_cost + overlap_cost

        return comm_cost

    def _mem_cost(self, i, j, model_p_element):
        if i == 0:
            if j == 0:
                # 不切P
                return self.dtype_size * model_p_element
            else:
                # 对P切j*8份
                return self.dtype_size * model_p_element / (j * 8)
        elif i == 1:
            if j == 0:
                # 不切G
                return self.dtype_size * model_p_element
            else:
                # 对G切j*8份
                return self.dtype_size * model_p_element / (j * 8)
        else:
            if j == 0:
                # 不切OS
                return self.dtype_size * self.fp32_ratio * 3 * model_p_element
            else:
                # 对OS切j*8份
                return self.dtype_size * self.fp32_ratio * 3 * model_p_element / (j * 8)

    def _get_comm_cost(self, num_strategies, sp_size, overlap_res, model_p_element):
        # 通信开销
        # P、G、OS切多少份对应的通信开销
        C = [[0 for _ in range(num_strategies)] for _ in range(3)]
        for i in range(3):
            for j in range(num_strategies):
                # TODO：这里需要支持更多的切分策略
                C[i][j] = self._comm_cost(i, j, sp_size, overlap_res, model_p_element)
        return C

    def _get_mem_cost(self, num_strategies, model_p_element):
        # memory占用
        # P、G、OS切多少份对应的memory开销
        A = [[0 for _ in range(num_strategies)] for _ in range(3)]
        for i in range(3):
            for j in range(num_strategies):
                A[i][j] = self._mem_cost(i, j, model_p_element)
        return A

    def _get_os_comm_cost(self, comm_range, model_p_element):
        if comm_range <= 1:
            return 0
        comm_volume = self.dtype_size * model_p_element
        return get_comm_cost(SovlerType.OS, CostType.BROADCAST, comm_range, comm_volume)

    def dump_constraint(self, C, A, pp, sp, memory_threshold, activation, micro_bsz, micro_num):
        print(f"<<<<<<< pp:{pp}, sp:{sp} >>>>>>>>", flush=True)
        print(f"C: {C}", flush=True)
        print(f"A: {A}", flush=True)
        print(
            f"memory_threshold: {memory_threshold/ GB:.2f} GB, activation: {activation/ GB:.2f} GB, micro_bsz: {micro_bsz}, micro_num: {micro_num}",
            flush=True,
        )
        print(f"<<<<<<<                  >>>>>>>>", flush=True)

    def run_loop(self):
        min_cost = float("inf")
        min_cost_solution = None
        for pp in PPIter(self.world_size, self._l):
            for sp in SPIter(self.world_size // pp, self._a):
                bs_bns = self.get_bsz(pp, sp, self.seq_len)
                for micro_bsz, micro_num in bs_bns:
                    layer_nums = self._l // pp
                    seq_len = self.seq_len // sp
                    pp_model_p_element = self._param_elements // pp

                    if algo_type == AlgoType.ISP:
                        self.algo = ISP(config=config, cost_data=self.cost_data, model_config=model_cfg, X=self._X, C=self._C, A=self._A, num_strategies=self._num_strategies)
                    elif algo_type == AlgoType.MSP:
                        self.algo = MSP(config=config, cost_data=self.cost_data, model_config=model_cfg, X=self._X, C=self._C, A=self._A, num_strategies=self._num_strategies)
                    elif algo_type == AlgoType.FSP:
                        self.algo = FSP(config=config, cost_data=self.cost_data, model_config=model_cfg, X=self._X, C=self._C, A=self._A, num_strategies=self._num_strategies)

                    self._memory_threshold = self.algo.set_memory_threshold()
                    self.algo.get_comm_cost()
                    self.algo.get_mem_cost()
                    self._X, self._C, self._A = self.algo.get_XCA()


                    overlap_res = TransformerOverlap(
                        b=self.dtype_size,
                        s=seq_len,
                        h=self._h,
                        # a=self._a,
                        num_layers=layer_nums,
                        dtype_size=self.dtype_size,
                        mlp_ratio=self.mlp_ratio,
                        multiple_of=self.multiple_of,
                        vocab_size=self.vocab_size,
                        cost_data=self.cost_data,
                    )

                    num_strategies = int(log2(self.world_size / 8)) + 2
                    C = self._get_comm_cost(num_strategies, sp, overlap_res, self._param_elements)
                    A = self._get_mem_cost(num_strategies, pp_model_p_element)

                    memory_threshold, activation = self._get_memory_threshold(micro_bsz, sp, pp, layer_nums, seq_len)

                    self.dump_constraint(C, A, pp, sp, memory_threshold, activation, micro_bsz, micro_num)
                    simulator = Simulator(memory_threshold, num_strategies, C=C, A=A)
                    cost, solution, mem_cost = simulator.run()
                    print(f"min_cost: {cost}, solution:{solution}", flush=True)
                    if cost is not None and cost < min_cost:
                        min_cost = cost
                        min_cost_solution = (pp, sp, micro_bsz, micro_num, mem_cost, solution)

        if min_cost_solution is not None:
            print("Minimum Communication Cost:", min_cost)
            print("Solution:", min_cost_solution)
        else:
            print("No solution found")
