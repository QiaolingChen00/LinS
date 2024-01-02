#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import math
import warnings

import torch
from einops import rearrange
from flash_attn.modules.mha import FlashSelfAttention, SelfAttention
from torch import nn

from profiler.registry import BENCHMARK_INITIALIZER
from utils.common import K, get_local_rank

from .base_benchmark import UnitBench

BENCH_TYPE = "flash_attn"


class MHA(nn.Module):
    def __init__(self, head_dim, causal=True, dropout=False, use_flash_attn=True) -> None:
        super().__init__()
        softmax_scale = 1 / math.sqrt(head_dim)
        self.causal = causal
        self.head_dim = head_dim
        inner_attn_cls = FlashSelfAttention if use_flash_attn else SelfAttention
        self.inner_attn = inner_attn_cls(causal=self.causal, softmax_scale=softmax_scale, attention_dropout=dropout)

    def forward(self, qkv, cu_seqlens, max_seqlen):
        # qkv = self.Wqkv(x)  # total x hsz'
        qkv = rearrange(qkv, "t (three h d) -> t three h d", three=3, d=self.head_dim)  # total x 3 x n_head x d
        context = self.inner_attn(qkv, causal=self.causal, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

        context = rearrange(context, "b h d -> b (h d)")  # recover the shape
        # out = self.out_proj(context)


@BENCHMARK_INITIALIZER.register_module(module_name=BENCH_TYPE)
class UnitMultiHeadAttn(UnitBench):
    test_loop = {
        "seq_len": [256 * K, int(0.25 * K), int(0.5 * K), 1 * K, 2 * K, 4 * K, 8 * K, 16 * K, 32 * K, 64 * K, 128 * K],
        "num_heads_and_hidden_dim": [(80, 10240), (64, 8192), (48, 6144), (40, 5120), (32, 4096)],  #
        # "num_heads_and_hidden_dim": [(80, 10240)],  # , 256 * K
        "dtype": [torch.bfloat16],
        "micro_bsz": [1, 2, 4, 8],
        "tp_size": [1, 2, 4, 8, 16, 32, 64],
    }

    # test_loop = {
    #     "seq_len": [8 * K],
    #     "num_heads_and_hidden_dim": [(40, 5120)],  #
    #     # "num_heads_and_hidden_dim": [(80, 10240)],  # , 256 * K
    #     "dtype": [torch.bfloat16],
    #     "micro_bsz": [1, 2],
    #     "tp_size": [2],
    # }

    def __init__(self, seq_len, num_heads_and_hidden_dim, dtype, micro_bsz, tp_size) -> None:
        num_heads, embed_dim = num_heads_and_hidden_dim
        self.tp_size = tp_size
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.dtype = dtype
        self.dtype_size = 2 if self.dtype == torch.bfloat16 else 4
        self.micro_bsz = micro_bsz
        self.oom = False

        assert self.embed_dim % self.tp_size == 0
        assert self.embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        assert num_heads % self.tp_size == 0, "num_heads must be divisible by tp_size"
        assert num_heads >= tp_size, f"head nums must bigger then tp_size: {tp_size}"

        self.num_atten_head_tp = num_heads // self.tp_size
        self.head_dim = self.embed_dim // num_heads

        self.packed_length = self.seq_len * self.micro_bsz
        self.device = f"cuda:{get_local_rank()}"

        indexs, cu_seqlens = [], [0]
        cu_seqlens = [i * self.seq_len for i in range(self.micro_bsz + 1)]
        indexs = list(range(self.seq_len)) * self.micro_bsz

        weights_mem_used = self.packed_length * 3 * self.embed_dim * self.dtype_size
        attn_activation = 11 * self.packed_length * self.embed_dim
        mem_used = attn_activation + weights_mem_used

        oom = False
        if mem_used > 75 * 1024**3:
            oom = True
        if self.packed_length >= 256 * K:  # and self.embed_dim / self.tp_size >= 6144:
            oom = True

        if oom:
            assert (
                False
            ), f"warning : mem_used: {mem_used/1024**3:.2f} GB, seq_len: {self.seq_len}, embed_dim: {self.embed_dim}, tp_size: {self.tp_size}"

        self.qkv = torch.rand(
            size=(self.packed_length, 3 * self.num_atten_head_tp * self.head_dim),
            dtype=self.dtype,
            device=self.device,
        )

        self.dtype_size = self.qkv.element_size()
        self.indexs = torch.tensor(data=indexs, dtype=torch.int32, device=self.device)
        self.cu_seqlens = torch.tensor(data=cu_seqlens, dtype=torch.int32, device=self.device)
        self.MHA = MHA(head_dim=self.head_dim, causal=True)

    def run(self):
        self.MHA(self.qkv, cu_seqlens=self.cu_seqlens, max_seqlen=self.seq_len)

    @staticmethod
    def gen_store_key(micro_bsz, seq_len, embed_dim, num_heads, tp_size):
        return f"b_{micro_bsz}_s_{seq_len}_h_{embed_dim}_a_{num_heads}_tp_{tp_size}"

    def complexity(self):
        return UnitMultiHeadAttn.gen_store_key(
            self.micro_bsz, self.seq_len, self.embed_dim, self.num_heads, self.tp_size
        )
        # return f"{self.seq_len} * {self.hidden_dim} * {self.hidden_dim}"
