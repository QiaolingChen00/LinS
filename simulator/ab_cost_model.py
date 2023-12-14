from utils.common import BW, CostType, SovlerType


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


def intra_inter_BW(solver_type, n, parallel_config=None):
    if n > 8:
        return BW.IB
    elif n == 1:
        return 10**9  # nearly zero
    else:
        if solver_type == SovlerType.MODEL:
            return BW.A800_NVL
        elif solver_type == SovlerType.PP:
            return BW.IB
        elif solver_type == SovlerType.OS:
            return BW.IB

    raise ValueError(f"unkonw solver_type: {solver_type}")


def get_comm_cost(solver_type, comm_op, comm_range, comm_volume, parallel_config=None):
    return coll_algo_bw(comm_op, comm_volume, comm_range) / intra_inter_BW(solver_type, comm_range)
