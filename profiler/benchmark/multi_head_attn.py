#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import math
import warnings

import torch
from einops import rearrange
from torch import nn

# from flash_attn.modules.mha import FlashSelfAttention, SelfAttention
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
        "seq_len": [int(0.5 * K), 1 * K, 2 * K, 4 * K, 8 * K, 16 * K, 32 * K],
        "num_heads_and_hidden_dim": [(32, 4096), (40, 5120), (48, 6144), (64, 8192), (80, 10240)],
        "dtype": [torch.bfloat16],
        "micro_bsz": [1, 2, 4, 8, 16],
    }

    def __init__(self, seq_len, micro_bsz, num_heads_and_hidden_dim, dtype) -> None:
        num_heads, embed_dim = num_heads_and_hidden_dim
        self.sp_size = 1
        self.tp_size = 1
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.dtype = dtype
        self.seq_len_sp = self.seq_len // self.sp_size
        self.num_attn_head_tp = self.num_heads // self.tp_size
        self.head_dim = self.embed_dim // self.num_heads

        self.micro_bsz = micro_bsz
        self.packed_length = self.seq_len_sp * self.micro_bsz

        self.max_seqlen = self.seq_len_sp  # maybe?
        self.seq_nums = max(self.seq_len, self.max_seqlen) // self.max_seqlen
        self.device = f"cuda:{get_local_rank()}"

        indexs, cu_seqlens = [], [0]
        left_tokens = self.packed_length
        for seq in range(self.seq_nums):
            ll = range(min(left_tokens, self.max_seqlen))
            indexs.append(list(ll))
            cu_seqlens.append(len(ll))
            left_tokens -= self.max_seqlen

        self.qkv = torch.rand(
            size=(self.packed_length, 3 * self.num_attn_head_tp * self.head_dim), dtype=self.dtype, device=self.device
        )
        self.dtype_size = self.qkv.element_size()
        self.indexs = torch.tensor(data=indexs, dtype=torch.int32, device=self.device)
        self.cu_seqlens = torch.tensor(data=cu_seqlens, dtype=torch.int32, device=self.device)

        self.MHA = MHA(head_dim=self.head_dim, causal=True)

    def run(self):
        self.MHA(self.qkv, cu_seqlens=self.cu_seqlens, max_seqlen=self.max_seqlen)

    @staticmethod
    def gen_store_key(micro_bsz, seq_len, embed_dim):
        return f"b_{micro_bsz}_s_{seq_len}_e_{embed_dim}"

    def complexity(self):
        return UnitMultiHeadAttn.gen_store_key(self.micro_bsz, self.seq_len, self.embed_dim)
        # return f"{self.seq_len} * {self.hidden_dim} * {self.hidden_dim}"
