import torch

"""Copyright The Microsoft DeepSpeed Team"""

import functools
import inspect
import os
import sys
import time
from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List

import torch
import torch.distributed as dist

from utils.common import OUT_OF_MEM_LATENCY, get_global_rank, get_world_size, sync_all

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
            try:
                test_case.run()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                return OUT_OF_MEM_LATENCY
        try:
            sync_all()
        except RuntimeError:
            #  self.packed_length * 3 * self.embed_dim * self.dtype_size
            print(
                f"packed_length: {test_case.packed_length}, embed_dim: {test_case.embed_dim}, micro_bsz: {test_case.micro_bsz}, seq_len: {test_case.seq_len}",
                flush=True,
            )
            torch.cuda.empty_cache()
            return OUT_OF_MEM_LATENCY

        # time the actual comm op trials times and average it
        pre = time.perf_counter()
        for _ in range(args.trials):
            try:
                test_case.run()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                return OUT_OF_MEM_LATENCY
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
        world_size = get_world_size()
        if world_size not in re_results:
            re_results[world_size] = {}
        try:
            bench = BENCH(**filter_kwargs(BENCH.__init__, test_case))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            test_case["lat"] = OUT_OF_MEM_LATENCY
        else:
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
