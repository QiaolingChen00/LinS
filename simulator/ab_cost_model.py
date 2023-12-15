from simulator.context import ParallelMode
from simulator.context import global_context as gpc
from utils.common import BW, CostType


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


def get_comm_cost(parallel_mode: ParallelMode, comm_op, comm_range, comm_volume, parallel_config=None):
    return coll_algo_bw(comm_op, comm_volume, comm_range) / gpc.check_pg_is_intra(parallel_mode)
