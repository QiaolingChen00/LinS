from z3 import *
from math import log2
from comm import TransformerCommunication
class Simulator:
    def __init__(self,config:dict) -> None:
        self._world_size=config["world_size"]
        self._global_batch_size=config["global_batch_size"]
        self._sequence_length=config["sequence_length"]
        self._model_size=config["model_size"]
        self._grad_acc=config["grad_acc"]
        self._SP=config["SP"]
        self._micro_batch_size=config["micro_bs"]
        print(config)
        self._set_memory_threshold()

        self._num_strategies = int(log2(self._world_size / 8)) + 2
        self._X = [[Bool(f"X_{i}_{j}") for j in range(self._num_strategies)] for i in range(3)]
        self._C = [[0 for _ in range(self._num_strategies)] for _ in range(3)]
        self._A= [[0 for _ in range(self._num_strategies)] for _ in range(3)]
        self._get_comm_cost()
        self._get_mem_cost()

        self._solver = Solver()

    def _get_model_config(self):
        if self._model_size==7:
            self._h=4096
            self._a=32
            self._l=32
        elif self._model_size==13:
            self._h=5120
            self._a=40
            self._l=40
        elif self._model_size==30:
            self._h=6144
            self._a=48
            self._l=60
        else: 
            self._h=8192
            self._a=64
            self._l=80

        return self._h,self._a,self._l

    def _set_memory_threshold(self):

        self._h,self._a,self._l=self._get_model_config()
        self._activation=self._micro_batch_size*self._sequence_length*self._h*(34+(5*self._a*self._sequence_length/self._h))/10**9/self._SP

        self._memory_threshold = 80 - self._activation
        print(f'micro_batch_size{self._micro_batch_size},activation{self._activation}')

    def _comm_cost(self,i,j):
        self._SP_comm_num=4*self._micro_batch_size*self._h*self._sequence_length/1024/1024/self._SP
        self._SP_comm=self._get_sp_comm_cost(self._SP_comm_num)
        comm_cost=0
        if j==0:
            comm_cost=0
        if i == 0:
            comm_cost=2 * self._model_size * (j + 1) 
        elif i == 1:
            comm_cost=self._grad_acc * self._model_size * (j + 1)
        else:
            comm_cost=self._model_size * (j + 1)

        comm_cost=comm_cost+self._SP_comm

        return comm_cost
    def _mem_cost(self,i,j):
        if i == 0:
            if j==0:
                return 2 * self._model_size
            return 2 * self._model_size / (j*8)
        elif i == 1:
            if j==0:
                return 2*self._model_size
            return 2 * self._model_size / (j * 8)
        else:
            if j==0:
                return 12*self._model_size
            return 12*self._model_size /(j *8)  
        
    def _get_comm_cost(self):
        for i in range(3):
            for j in range(self._num_strategies):
                self._C[i][j] = self._comm_cost(i, j)   
    def _get_mem_cost(self):
        for i in range(3):
            for j in range(self._num_strategies):
                self._A[i][j] = self._mem_cost(i, j)

    def _get_sp_comm_cost(self,_SP_comm_num):
        if self._SP==1:
            return 0
        dis=[1,0.9,0.8,0.6,0.1]
        sp_comm_cost=_SP_comm_num/dis[int(log2(self._SP))]
 
        return sp_comm_cost 


    def _strategy_constraint_strict(self):
        for i in range(3):
            self._solver.add(Sum([If(self._X[i][j], 1, 0) for j in range(self._num_strategies)]) == 1)
        for j in range(1, self._num_strategies):  
            self._solver.add(Implies(self._X[0][j], Not(self._X[1][j-1])))
        for j in range(1, self._num_strategies):  
            self._solver.add(Implies(self._X[1][j], Not(self._X[2][j-1])))

            
    def _memory_constraint_strict(self):
        total_memory = Sum([self._A[i][j] * If(self._X[i][j], 1, 0) for i in range(3) for j in range(self._num_strategies)])
        self._solver.add(total_memory < self._memory_threshold)
    def _build_constraint(self):
        self._strategy_constraint_strict()
        self._memory_constraint_strict()
    def _build_optimize_object(self):
        self._total_comm_cost = Int('total_comm_cost')
        self._communication_cost_expr = Sum([self._C[i][j] * If(self._X[i][j], 1, 0) for i in range(3) for j in range(self._num_strategies)])
        self._solver.add(self._total_comm_cost == self._communication_cost_expr)
    def run(self):
        self._build_constraint()
        self._build_optimize_object()
   
        self._solver.push()
        min_cost = None
        while self._solver.check() == sat:
            model = self._solver.model()
            
            current_cost = model[self._total_comm_cost].as_long()
            if min_cost is None or current_cost < min_cost:
                min_cost = current_cost
                solution = [[model.evaluate(self._X[i][j]) for j in range(self._num_strategies)] for i in range(3)]
            self._solver.add(self._total_comm_cost < current_cost)  # Add constraint to find lower cost

        if min_cost is not None:
            print("Minimum Communication Cost:", min_cost)
            print("Solution:", solution)
        else:
            print("No solution found")

