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
            
        # obtain the model cofing, hidden_size, number of heads, layer num
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
        
        # set the memory threshold
        self._set_memory_threshold()
        
        # TODO 需要支持更多的切分策略
        self._num_strategies = int(log2(self._world_size / 8)) + 2
        
        # 并行策略的组合选项
        # i: P，G，OS； j：节点数
        self._X = [[Bool(f"X_{i}_{j}") for j in range(self._num_strategies)] for i in range(3)]
        # 通信开销
        # P、G、OS切多少份对应的通信开销
        self._C = [[0 for _ in range(self._num_strategies)] for _ in range(3)]
        # memory占用
        # P、G、OS切多少份对应的memory开销
        self._A = [[0 for _ in range(self._num_strategies)] for _ in range(3)]
        
        self._get_comm_cost()
        
        self._get_mem_cost()
        
        print(f'self._X{self._X}')
        print(f'self._C{self._C}')
        print(f'self._A{self._A}')


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
            / self._SP
        )
        # 或许这里还少了一点东西？比如os? os在forward-backward和step的占比不一样
        self._memory_threshold = 80 * (1024 ** 3) - self._activation
        if self._memory_threshold < 0:
            print(f"!!!warning!!!: self._memory_threshold: {self._memory_threshold} < 0")
        print(f"activation: {self._activation:.4f} GB")

    def _lookup_comm_cost(self, type: CostType, world_size, complexity):
        return self.cost_data[type].predict(world_size, complexity)

    # 这个通信量的计算是包括forward+backward？
    def _comm_cost(self, i: int, j: int) -> float:
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
        # self._SP_comm = self._get_sp_comm_cost(self._SP)

        if j != 0:
            comm_range = j * 8
        else:
            comm_range = 1  # no comm cost

        # 算overlap的通信开销
        overlap_cost = self.overlap_res._get_overlap(comm_range, self._SP)

        
        # 算os的通信开销
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

        # 总的通信开销
        comm_cost = os_comm_cost + overlap_cost

        return comm_cost

    def _mem_cost(self, i, j):
        if i == 0:
            if j == 0:
                # 不切P
                return self._dtype_size * self._model_size
            # 对P切j*8份
            return self._dtype_size * self._model_size / (j * 8)
        elif i == 1:
            if j == 0:
                # 不切G
                return self._dtype_size * self._model_size
            # 对G切j*8份
            return self._dtype_size * self._model_size / (j * 8)
        else:
            if j == 0:
                # 不切OS
                return self._dtype_size * self._os_size_ratio * 3 * self._model_size
            # 对OS切j*8份
            return self._dtype_size * self._os_size_ratio * 3 * self._model_size / (j * 8)

    def _get_comm_cost(self):
        for i in range(3):
            for j in range(self._num_strategies):
                #TODO：这里需要支持更多的切分策略
                if j != 1 and j % 2 != 0: # 节点数为奇数的时候
                    self._C[i][j] = self._C[i][j - 1] * 1.2
                else: # 节点数为偶数
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
        # TODO：这里是否要全量的通信数据？
        comm_cost = self._dtype_size * self._p_size
        return self._lookup_comm_cost(CostType.ALLGATHER, comm_range, comm_cost)  # TODO: Should be broadcast

    def _strategy_constraint_strict(self):
        for i in range(3): #xyt: 对于P、G、OS只可能有一种切分策略
            self._solver.add(Sum([If(self._X[i][j], 1, 0) for j in range(self._num_strategies)]) == 1)
        #TODO：如果要让P和G切分保持一致，下面这个限制可能需要修改
        for j in range(1, self._num_strategies): #xyt:感觉这里是想说明P和G的切分是绑定在一起的
            self._solver.add(Implies(self._X[0][j], Not(self._X[1][j - 1])))
        for j in range(1, self._num_strategies): 
            self._solver.add(Implies(self._X[1][j], Not(self._X[2][j - 1])))

    def _memory_constraint_strict(self):
        #TODO: 可能有两个限制，一个是forward+backward；一个是step，这两个阶段都需满足显存不超过80GB
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
