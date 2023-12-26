import math
import os

import torch
import torch.distributed as dist
from torch.distributed import GroupMember


# TODO: 这里需要增加一个broadcast
class CostType:
    ALL2ALL = "all2all"
    ALLREDUCE = "all_reduce"
    REDUCESCATTER = "reduce_scatter"
    ALLGATHER = "all_gahter"
    LINEAR = "linear"
    BROADCAST = "broadcast"
    P2P = "p2p"
    FLASH_ATTN = "flash_attn"


class AlgoType:
    ISP = "isp"
    MSP = "msp"
    FSP = "fsp"
    MTP = "mtp"
    NONE = "none"


class BW:
    IB = 100 * 1024**3
    A800_NVL = 200 * 1024**3
    A100_NVL = 300 * 1024**3


BENCH_TYPE_LIST = [CostType.ALL2ALL, CostType.ALLREDUCE, CostType.REDUCESCATTER, CostType.ALLGATHER, CostType.LINEAR]

# BENCH_TYPE_LIST = [CostType.ALL2ALL, CostType.ALLREDUCE, CostType.REDUCESCATTER, CostType.ALLGATHER, CostType.LINEAR]

K = 1024

KB = 1024
MB = 1024 * KB
GB = 1024 * MB

MS = 1000
US = 1000 * MS

_75GB = 75 * GB
_100GB = 100 * GB


OUT_OF_MEM_LATENCY = 10**9


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
    elif model_size == 65:
        h = 8192
        a = 64
        l = 80
    else:
        raise ValueError(f"unsupport modesize: {model_size}")

    mlp_ratio = 8 / 3
    multiple_of = 256

    return h, a, l, mlp_ratio, multiple_of


def pretty_print_size(x):
    if x < KB:
        return f"{x} B"
    elif x >= KB and x < MB:
        return f"{x/KB:.3f} KB"
    elif x >= MB and x < GB:
        return f"{x/MB:.3f} MB"
    else:
        return f"{x/GB:.3f} GB"


def pretty_print_latency(x):
    if x >= 1:
        return f"{x:.3f} s"
    elif x >= 1 / MS and x < 1:
        return f"{x*MS:.3f} ms"
    else:
        return f"{x*US:.3f} us"


def get_local_rank():
    if "SLURM_PROCID" in os.environ:
        return int(os.environ["SLURM_PROCID"]) % 8
    else:
        return 0


def get_world_size():
    if "SLURM_NPROCS" in os.environ:
        return int(os.environ["SLURM_NPROCS"]) % 8
    else:
        return 1


def sync_all():
    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()


def get_bw(comm_op, size, duration, args):
    n = dist.get_world_size()
    tput = 0
    busbw = 0
    if comm_op == "all_to_all":
        tput = size / duration
        busbw = (size / duration) * ((n - 1) / n)
    elif comm_op == "all_gather" or comm_op == "reduce_scatter":
        size *= n
        tput = size / duration
        busbw = (size / duration) * ((n - 1) / n)
    elif comm_op == "all_reduce":
        tput = size * 2 / duration
        busbw = (size / duration) * (2 * (n - 1) / n)
    elif comm_op == "pt2pt" or comm_op == "broadcast":
        tput = size / duration
        busbw = tput
    else:
        print("wrong comm_op specified")
        exit(0)

    if args.bw_unit == "Gbps":
        tput *= 8
        busbw *= 8

    return tput, busbw


sub_process_groups = {}
TORCH_DISTRIBUTED_DEFAULT_PORT = 12349


def env2int(env_list, default=-1):
    for e in env_list:
        val = int(os.environ.get(e, -1))
        if val >= 0:
            return val
    return default


def init_torch_distributed(backend):
    global dist
    import torch.distributed as dist

    # discover rank/size info from env
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = str(TORCH_DISTRIBUTED_DEFAULT_PORT)
    if "MASTER_ADDR" not in os.environ:
        import subprocess

        result = subprocess.check_output('scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1', shell=True)
        master_addr = result.decode("utf8").strip()
        if master_addr == "":
            master_addr = "127.0.0.1"
        os.environ["MASTER_ADDR"] = master_addr
    local_rank = env2int(
        ["LOCAL_RANK", "MPI_LOCALRANKID", "OMPI_COMM_WORLD_LOCAL_RANK", "MV2_COMM_WORLD_LOCAL_RANK", "SLURM_LOCALID"]
    )
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(local_rank)
    rank = env2int(["RANK", "MPI_RANKID", "OMPI_COMM_WORLD_RANK", "MV2_COMM_WORLD_RANK", "SLURM_PROCID"])
    if "RANK" not in os.environ:
        os.environ["RANK"] = str(rank)
    world_size = env2int(["WORLD_SIZE", "OMPI_COMM_WORLD_SIZE", "MV2_COMM_WORLD_SIZE", "SLURM_NPROCS"])
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = str(world_size)

    torch.distributed.init_process_group(backend)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)


def build_process_gourp(max_world_size):
    global sub_process_groups
    if max_world_size > 1:
        init_torch_distributed("nccl")
        sub_process_groups[str(dist.get_world_size())] = GroupMember.WORLD

        if dist.is_initialized():
            world_size = dist.get_world_size()
            node_nums = world_size // 8
            base_num = [2, 4, 6]
            base_num = base_num + [8 * i for i in range(1, node_nums)]

            # if base_num <= 3:
            #     return
            for gpu_nums in base_num:
                ranks = [j for j in range(gpu_nums)]
                print(ranks, flush=True)
                sub_process_groups[f"{gpu_nums}"] = dist.new_group(ranks)
                # dist.get_process_group_ranks()


def get_global_rank():
    if dist.is_initialized():
        return dist.get_rank()
    else:
        return 0
