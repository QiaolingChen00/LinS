import os
import pickle

import torch.distributed as dist

import profiler.benchmark
from profiler.profiler import (
    print_bench_reulsts,
    reformat_data_to_cost_model,
    run_profile,
)
from utils.common import *
from utils.config import Config
from simulator.predict_cost_model import CostModel
# if __name__ == "__main__":
#     total_results = {
#         '32' : {
#             '1': [{'lat' : 0.01}],
#             '2': [{'lat' : 0.02}],
#             '3': [{'lat' : 0.03}],
#         },
#         '16' : {
#             '1': [{'lat' : 0.01}],
#             '2': [{'lat' : 0.02}],
#             '3': [{'lat' : 0.03}],
#         }
#     }
#     print(reformat_data_to_cost_model(total_results))


if __name__ == "__main__":
    world_size = int(os.environ['SLURM_NPROCS'])
    build_process_gourp(world_size)
    cost_model = CostModel(is_master=get_global_rank() == 0)
    cost_model.build_cost_model()
    # cost_model.dump_data()
    cost_model.dump_all_data()

