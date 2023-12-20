import functools

from simulator.context import ParallelMode
from simulator.context import global_context as gpc
from utils.common import BW, CostType

scale_ratio = [1.415134488, 1.208864145, 1.1, 1]


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


def get_scale_ratio(scale):
    # 通信扩展惩罚系数
    if scale <= 16:
        return 1
    elif scale > 16 and scale <= 32:
        return 1.1
    elif scale > 32 and scale <= 64:
        return 1.2
    elif scale > 64:
        return 1.4


def get_comm_cost(comm_volume: int, parallel_mode: ParallelMode, comm_op: CostType = None):
    """根据通信量获得近似的通信延迟,这个函数考虑了跨节点带宽content的情景
    所以为了正确计算延迟，传入的 comm_volume 必须是以单个rank视角下的通信量
    (即代码中实际传入的通信量)

    Args:
        comm_volume (int): 通信量, 单位B
        parallel_mode (ParallelMode): gpc并行模式
        comm_op (CostType, optional): 通信算子

    Returns:
        int: 通信延迟,是乘以10**4后并取整后的数值
    """
    scale = gpc.get_world_size(parallel_mode)

    if parallel_mode == ParallelMode.PIPELINE:
        scale = 2
    if scale <= 1:
        return 0

    is_intra = gpc.check_pg_is_intra(parallel_mode)
    if not is_intra:
        num_partner = gpc.same_group_in_one_node(parallel_mode)
        # if parallel_mode == ParallelMode.ZERO1:
        #     assert num_partner == 1
        comm_volume = comm_volume * num_partner

    bw = BW.A800_NVL if is_intra else (BW.IB / get_scale_ratio(scale))
    return int(1000 * 10 * coll_algo_bw(comm_op, comm_volume, scale) / bw)  # 转换成ms小数点保留两位


allgather = functools.partial(get_comm_cost, comm_op=CostType.ALLGATHER)
reducescatter = functools.partial(get_comm_cost, comm_op=CostType.REDUCESCATTER)
broadcast = functools.partial(get_comm_cost, comm_op=CostType.BROADCAST)
p2p = functools.partial(get_comm_cost, comm_op=CostType.P2P)
alltoall = functools.partial(get_comm_cost, comm_op=CostType.ALL2ALL)
allreduce = functools.partial(get_comm_cost, comm_op=CostType.ALLREDUCE)
