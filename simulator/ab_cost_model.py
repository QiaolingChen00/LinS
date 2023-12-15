from simulator.context import ParallelMode
from simulator.context import global_context as gpc
from utils.common import BW, CostType
import functools


def coll_algo_bw(comm_op, size, n):
    if comm_op == CostType.ALL2ALL:
        return size * (n - 1) / n
    elif comm_op == CostType.ALLREDUCE:
        return size * 2 * (n - 1) / n
    elif comm_op == CostType.REDUCESCATTER:
        return size * (n - 1) / n
    elif comm_op == CostType.ALLGATHER:
        return size * (n - 1) / n
    elif comm_op == CostType.BROADCAST:
        return size * (n - 1) / n
    elif comm_op == CostType.P2P:
        return size

    raise ValueError(f"unkonw comm_op: {comm_op}")



def get_comm_cost(comm_volume: int, parallel_mode: ParallelMode,comm_op: CostType = None):
    scale = gpc.get_world_size(parallel_mode)
    if scale <= 1:
        return 0
    
    is_intra = gpc.check_pg_is_intra(parallel_mode)
    bw = BW.A800_NVL if is_intra else BW.IB
    return int(1000 * 10 * coll_algo_bw(comm_op, comm_volume, scale) / bw)   # 转换成ms小数点保留两位

allgather = functools.partial(get_comm_cost, comm_op=CostType.ALLGATHER)
reducescatter = functools.partial(get_comm_cost, comm_op=CostType.REDUCESCATTER)
broadcast = functools.partial(get_comm_cost, comm_op=CostType.BROADCAST)
p2p = functools.partial(get_comm_cost, comm_op=CostType.P2P)
alltoall = functools.partial(get_comm_cost, comm_op=CostType.ALL2ALL)
allreduce = functools.partial(get_comm_cost, comm_op=CostType.ALLREDUCE)
