import os
import pickle

import torch.distributed as dist

import profiler.benchmark
from cost_model import PolynomialModel
from profiler.profiler import (
    print_bench_reulsts,
    reformat_data_to_cost_model,
    run_profile,
)
from utils.common import *
from utils.config import Config

bench_type_list = ["linear", "all_reduce", "all_gahter", "all2all", "reduce_scatter"]


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

    args = Config(
        {
            "trials": 10,
            "warmups": 5,
        }
    )

    build_process_gourp(64)

    for BENCH_TYPE in bench_type_list:
        if get_global_rank() == 0:
            print(f"now test {BENCH_TYPE}", flush=True)
        dump_file = f"./data/dump_data_{BENCH_TYPE}.pickle"

        if not os.path.exists(dump_file):
            re_results = run_profile(args, BENCH_TYPE)
            print_bench_reulsts(re_results)
            data = reformat_data_to_cost_model(re_results)
            if get_global_rank() == 0:
                with open(dump_file, "wb") as f:
                    pickle.dump(data, f)
        else:
            with open(dump_file, "rb") as f:
                data = pickle.load(f)

        if get_global_rank() == 0:
            print(data)
            linear_model = PolynomialModel(degree=2, data=data, name=f"./data/{BENCH_TYPE}")
            linear_model.build_model()
