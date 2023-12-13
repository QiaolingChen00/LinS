import math
import pickle
from math import log2
from z3 import *

from simulator.algo import ISP, MSP, FSP

# from comm import TransformerCommunication
# from utils.utils import _get_model_config
from utils.common import *
from utils.common import AlgoType



class Simulator:
    def __init__(self, config: dict, cost_data: dict = None, cost_data_path: str = None, algo_type: AlgoType = AlgoType.ISP) -> None:
        self._world_size = config["world_size"]
        self._model_size = config["model_size"]
  
        # obtain the cost_data model.
        if cost_data is None:
            if cost_data_path is not None:
                with open(cost_data_path, "rb") as f:
                    self.cost_data = pickle.load(f)
            else:
                self.cost_data = None
        else:
            self.cost_data = cost_data
            
        # obtain the model cofing, hidden_size, number of heads, layer num
        model_cfg = self._get_model_config()
        
        # TODO 需要支持更多的切分策略
        self._num_strategies = int(log2(self._world_size / 8)) + 2
        
        # 并行策略的组合选项
        # i: P，G，OS； j：节点数
        self._X = [[bool(f"X_{i}_{j}") for j in range(self._num_strategies)] for i in range(3)]
        # 通信开销
        # P、G、OS切多少份对应的通信开销
        self._C = [[0 for _ in range(self._num_strategies)] for _ in range(3)]
        # memory占用
        # P、G、OS切多少份对应的memory开销
        self._A = [[0 for _ in range(self._num_strategies)] for _ in range(3)]
        
        self.algo = None
        
        if algo_type == AlgoType.ISP:
            self.algo = ISP(config=config, cost_data=self.cost_data, model_config=model_cfg, X=self._X, C=self._C, A=self._A, num_strategies=self._num_strategies)
        elif algo_type == AlgoType.MSP:
            self.algo = MSP(config=config, cost_data=self.cost_data, model_config=model_cfg, X=self._X, C=self._C, A=self._A, num_strategies=self._num_strategies)
        elif algo_type == AlgoType.FSP:
            self.algo = FSP(config=config, cost_data=self.cost_data, model_config=model_cfg, X=self._X, C=self._C, A=self._A, num_strategies=self._num_strategies)
        
        # set the memory threshold
        # self._set_memory_threshold()
        self._memory_threshold = self.algo.set_memory_threshold()
        
        # get the communication cost
        # self._get_comm_cost()
        self.algo.get_comm_cost()
        
        # get the memory cost
        # self._get_mem_cost()
        self.algo.get_mem_cost()
        
        self._X, self._C, self._A = self.algo.get_XCA()
        
        print(f'self._X{self._X}')
        print(f'self._C{self._C}')
        print(f'self._A{self._A}')

        # solve!
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
        
        self.mlp_ratio = 8 / 3
        self.multiple_of = 256
        
        model_cfg = {}
        
        model_cfg["h"] = self._h
        model_cfg["a"] = self._a
        model_cfg["l"] = self._l
        model_cfg["mlp_ratio"] = self.mlp_ratio
        model_cfg["multiple_of"] = self.multiple_of

        return model_cfg

    def _strategy_constraint_strict(self):
        for i in range(3): #xyt: 对于P、G、OS只可能有一种切分策略
            self._solver.add(Sum([If(self._X[i][j], 1, 0) for j in range(self._num_strategies)]) == 1)
        for j in range(1, self._num_strategies): #xyt:感觉这里是想说明P和G的切分是绑定在一起的
            self._solver.add(self._X[0][j] == self._X[1][j])
        for j in range(1, self._num_strategies): # 这个约束条件保证os的切分不能小于G的切分
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
