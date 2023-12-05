import torch

"""Copyright The Microsoft DeepSpeed Team"""

import inspect
import os
import sys
import time
from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List

import torch
import torch.distributed as dist

from utils.common import *

from .registry import BENCHMARK_INITIALIZER


def DFS(loop_config: OrderedDict, results: OrderedDict, total_results: List):
    if len(loop_config) == 0:
        total_results.append(deepcopy(results))
        return

    now_key = list(loop_config.keys())[0]
    now_values = loop_config[now_key]
    loop_config.pop(now_key)

    for value in now_values:
        results.update({now_key: value})
        DFS(loop_config, results, total_results)

    loop_config[now_key] = now_values


def filter_kwargs(func, kwargs):
    sig = inspect.signature(func)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def run_profile(args, test_type):
    re_results = {}

    BENCH = BENCHMARK_INITIALIZER.get_module(module_name=test_type)

    def run_benchmark(test_case, args):
        sync_all()
        # Warmups, establish connections, etc.
        for _ in range(args.warmups):
            test_case.run()
        sync_all()

        # time the actual comm op trials times and average it
        pre = time.perf_counter()
        for _ in range(args.trials):
            test_case.run()
        sync_all()
        duration = time.perf_counter() - pre

        # maintain and clean performance data
        avg_duration = duration / args.trials
        return avg_duration

    sync_all()
    # loop over various tensor sizes
    test_args = OrderedDict(BENCH.test_loop)
    total_cases = []

    DFS(test_args, OrderedDict(), total_cases)
    if get_global_rank() == 0:
        print(f"all test case nums: {len(total_cases)}", flush=True)

    for test_case in total_cases:
        world_size = test_case["world_size"]
        if world_size not in re_results:
            re_results[world_size] = {}

        bench = BENCH(**filter_kwargs(BENCH.__init__, test_case))
        sync_all()
        avg_duration = run_benchmark(bench, args)
        test_case["lat"] = avg_duration

        # assert bench.complexity() not in re_results
        # re_results[bench.complexity()] = test_case
        if bench.complexity() not in re_results[world_size]:
            re_results[world_size][bench.complexity()] = [test_case]
        else:
            if get_global_rank() == 0:
                print(
                    f"Warning same complexity: {test_case['lat']:.3f} {re_results[world_size][bench.complexity()][0]['lat']:.5f}"
                )

    return re_results


def print_bench_reulsts(re_results: OrderedDict):
    pass


# def print_bench_reulsts(re_results: OrderedDict):
#     for world_size in re_results.keys():
#         for complexity in re_results[world_size].keys():
#             # print(f"complexity: {complexity}")
#             for test_case in re_results[complexity]:
#                 re = OrderedDict()
#                 re["complexity"] = complexity
#                 re.update(test_case)
#                 re["lat"] = pretty_print_latency(test_case["lat"])
#                 if get_global_rank() == 0:
#                     print(re)

import functools


def my_compare(a, b):
    world_size_a, complexity_a = a[0], a[2]
    world_size_b, complexity_b = b[0], b[2]
    # print(world_size_a, world_size_b, complexity_a, complexity_b)

    if world_size_a > world_size_b:
        return True
    elif world_size_a < world_size_b:
        return False
    else:
        if complexity_a > complexity_b:
            return True
        elif complexity_a < complexity_b:
            return False
        else:
            assert ValueError, f"a:{a}, b:{b}"


def reformat_data_to_cost_model(total_results):
    list_data = []

    for world_size in total_results.keys():
        for complexity in total_results[world_size].keys():
            for value in total_results[world_size][complexity]:
                print(value)
                list_data.append([world_size, value["lat"], complexity])

    list_data.sort(key=functools.cmp_to_key(my_compare))
    data_list = list(map(list, zip(*list_data)))
    data = {"World_Size": data_list[0], "Latency_ms": data_list[1], "Data_MB": data_list[2]}

    return data
