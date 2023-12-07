import math
import pickle
from math import log2

from z3 import *

from simulator.overlap import TransformerOverlap

# from comm import TransformerCommunication
# from utils.utils import _get_model_config
from utils.common import *


class Simulator:
    def __init__(self, config: dict, cost_data: dict = None, cost_data_path: str = None) -> None:
        self._world_size = config["world_size"]
        self._global_batch_size = config["global_batch_size"]
        self._sequence_length = config["sequence_length"]
        self._model_size = config["model_size"]
        self._grad_acc = config["grad_acc"]
        self._SP = config["SP"]
        self._micro_batch_size = config["micro_bs"]
        self._vocab_size = config["vocab_size"]
        self._dtype_size = 2
        self._os_size_ratio = 2 if self._dtype_size == 2 else 1
        self._p_size = self._model_size * 10**9
  

        if cost_data is None:
            if cost_data_path is not None:
                with open(cost_data_path, "rb") as f:
                    self.cost_data = pickle.load(f)
            else:
                self.cost_data = None
        else:
            self.cost_data = cost_data

        self._h, self._a, self._l = self._get_model_config()
        self.overlap_res = TransformerOverlap(
            b=self._dtype_size,
            s=self._sequence_length,
            h=self._h,
            # a=self._a,
            num_layers=self._l,
            vocab_size=self._vocab_size,
            cost_data=self.cost_data,
        )

        self._set_memory_threshold()
        self._num_strategies = int(log2(self._world_size / 8)) + 2
        self._X = [[Bool(f"X_{i}_{j}") for j in range(self._num_strategies)] for i in range(3)]
        self._C = [[0 for _ in range(self._num_strategies)] for _ in range(3)]
        self._A = [[0 for _ in range(self._num_strategies)] for _ in range(3)]
        self._get_comm_cost()
        self._get_mem_cost()


        self._solver = Solver()

    def _get_model_config(self):
        if self._model_size == 7:
            self._h = 4096
            self._a = 32
            self._l = 32
        elif self._model_size == 13:
            self._h = 5120
            self._a = 40
            self._l = 40
        elif self._model_size == 20:
            self._h = 5120
            self._a = 40
            self._l = 60
        elif self._model_size == 30:
            self._h = 6144
            self._a = 48
            self._l = 60
        else:
            self._h = 8192
            self._a = 64
            self._l = 80

        return self._h, self._a, self._l

    def _set_memory_threshold(self):
        self._activation = (
            self._dtype_size
            * self._micro_batch_size
            * self._sequence_length
            * self._h
            * (34 + (5 * self._a * self._sequence_length / self._h))
            / 10**9
            / self._SP
        )

        self._memory_threshold = 80 - self._activation
        if self._memory_threshold < 0:
            print(f"!!!warning!!!: self._memory_threshold: {self._memory_threshold} < 0")
        print(f"activation: {self._activation:.4f} GB")

    def _lookup_comm_cost(self, type: CostType, world_size, complexity):
        return self.cost_data[type].predict(world_size, complexity)

    def _comm_cost(self, i: int, j: int) -> float:
        """
        Get communication cost.

        Args:
            i (int): p (i==0), g (i==1), os (i==2)
            j (int): node count

        Returns:
            float: communication cost
        """
        # self._SP_comm = self._get_sp_comm_cost(self._SP)

        if j != 0:
            comm_range = j * 8
        else:
            comm_range = 1  # no comm cost

        overlap_cost = self.overlap_res._get_overlap(comm_range, self._SP)

        
        if comm_range == 1:
            os_comm_cost = 0
        else:
            os_comm_cost = self._get_os_comm_cost(comm_range)
            # if i == 0:
            #     comm_cost = self._get_p_comm_cost(world_size)
            # elif i == 1:
            #     comm_cost = self._get_g_comm_cost(world_size)
            # if i == 2:  # only consider OS comm cost.
            #     os_comm_cost = self._get_os_comm_cost(comm_range)
            # else:
            #     os_comm_cost = 0

        # if os_comm_cost < 0 or overlap_cost < 0:
        #     import pdb;pdb.set_trace()
        #     raise ValueError

        comm_cost = os_comm_cost + overlap_cost

        return comm_cost * 100

    def _mem_cost(self, i, j):
        if i == 0:
            if j == 0:
                return self._dtype_size * self._model_size
            return self._dtype_size * self._model_size / (j * 8)
        elif i == 1:
            if j == 0:
                return self._dtype_size * self._model_size
            return self._dtype_size * self._model_size / (j * 8)
        else:
            if j == 0:
                return self._dtype_size * self._os_size_ratio * 3 * self._model_size
            return self._dtype_size * self._os_size_ratio * 3 * self._model_size / (j * 8)

    def _get_comm_cost(self):
        for i in range(3):
            for j in range(self._num_strategies):
                if j != 1 and j % 2 != 0:
                    self._C[i][j] = self._C[i][j - 1] * 1.2
                else:
                    self._C[i][j] = self._comm_cost(i, j)

    def _get_mem_cost(self):
        for i in range(3):
            for j in range(self._num_strategies):
                if j != 1 and j % 2 != 0:
                    self._A[i][j] = self._A[i][j - 1] * 0.8
                else:
                    self._A[i][j] = self._mem_cost(i, j)

    def _get_os_comm_cost(self, comm_range):
        if comm_range <= 1:
            return 0
        comm_cost = self._dtype_size * self._p_size
        return self._lookup_comm_cost(CostType.ALLGATHER, comm_range, comm_cost)  # TODO: Should be broadcast

    def _strategy_constraint_strict(self):
        for i in range(3):
            self._solver.add(Sum([If(self._X[i][j], 1, 0) for j in range(self._num_strategies)]) == 1)
        for j in range(1, self._num_strategies):
            self._solver.add(Implies(self._X[0][j], Not(self._X[1][j - 1])))
        for j in range(1, self._num_strategies):
            self._solver.add(Implies(self._X[1][j], Not(self._X[2][j - 1])))

    def _memory_constraint_strict(self):
        total_memory = Sum(
            [self._A[i][j] * If(self._X[i][j], 1, 0) for i in range(3) for j in range(self._num_strategies)]
        )
        self._solver.add(total_memory < self._memory_threshold)

    def _build_constraint(self):
        self._strategy_constraint_strict()
        self._memory_constraint_strict()

    def _build_optimize_object(self):
        self._total_comm_cost = Real("total_comm_cost")
        self._communication_cost_expr = Sum(
            [self._C[i][j] * If(self._X[i][j], 1, 0) for i in range(3) for j in range(self._num_strategies)]
        )
        self._solver.add(self._total_comm_cost == self._communication_cost_expr)

    def run(self):
        self._build_constraint()
        self._build_optimize_object()

        self._solver.push()
        min_cost = None
        while self._solver.check() == sat:
            model = self._solver.model()

            # current_cost = model[self._total_comm_cost].as_long()
            current_cost_str = model[self._total_comm_cost].as_decimal(10)
            current_cost_str = current_cost_str.rstrip('?')
            current_cost_value = float(current_cost_str)  # 将字符串转换为浮点数
            if min_cost is None or current_cost_value < min_cost:
                min_cost = current_cost_value
                solution = [[model.evaluate(self._X[i][j]) for j in range(self._num_strategies)] for i in range(3)]
            # self._solver.add(self._total_comm_cost < current_cost)  # Add constraint to find lower cost
            self._solver.add(self._total_comm_cost < current_cost_value)


        if min_cost is not None:
            print("Minimum Communication Cost:", min_cost)
            print("Solution:", solution)
        else:
            print("No solution found")

    # def _get_g_comm_cost(self, comm_range): # discard: already count in _get_overlap
    #     if comm_range <= 1:
    #         return 0
    #     comm_cost = self._dtype_size * self._p_size * self._grad_acc
    #     return self._lookup_comm_cost(CostType.REDUCESCATTER, comm_range, comm_cost)

    # def _get_p_comm_cost(self, comm_range): # discard: already count in _get_overlap
    #     if comm_range <= 1:
    #         return 0
    #     comm_cost = 2 * self._dtype_size * self._p_size # bwd + fwd
    #     return self._lookup_comm_cost(CostType.ALLGATHER, comm_range, comm_cost)

    # def _get_sp_comm_cost(self, comm_range):    # discard: already count in _get_overlap
    #     if comm_range <= 1:
    #         return 0
    #     comm_cost = (
    #         self._dtype_size * 4 * self._micro_batch_size * self._h * self._sequence_length / 1024 / 1024 / self._SP
    #     )
    #     return self._lookup_comm_cost(CostType.ALL2ALL, comm_range, comm_cost)
