# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Packed, noncausal ViT attention using the shared varlen Triton kernel."""

import torch

from .fa_varlen_triton import flash_attn_varlen_triton


def flash_attn_fwd(q, k, v, *, cu_seqlens_q=None, cu_seqlens_k=None,
                   max_seqlen_q=None, max_seqlen_k=None, softmax_scale=None,
                   causal=False, window_size_left=-1, window_size_right=-1,
                   num_splits=1, pack_gqa=None, return_lse=False, **kwargs):
    if (kwargs or causal or window_size_left != -1 or window_size_right != -1
            or num_splits != 1 or pack_gqa not in (None, False)):
        raise NotImplementedError("Triton ViT attention requires the noncausal, full-window call")
    if (q.ndim != 3 or k.ndim != 3 or v.shape != k.shape
            or q.shape[1:] != k.shape[1:]):
        raise ValueError("ViT attention requires matching [tokens, heads, head_dim] Q/K/V")
    if not q.is_cuda or q.device != k.device or q.device != v.device:
        raise ValueError("ViT Q/K/V must share a GPU")
    if q.dtype not in (torch.bfloat16, torch.float16) or q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("Triton ViT attention requires matching BF16 or FP16 Q/K/V")
    dim = q.shape[-1]
    if dim not in (16, 32, 64, 128, 256) or q.shape[1] == 0:
        raise ValueError("Triton ViT attention requires nonempty heads of width 16, 32, 64, 128 or 256")
    if (cu_seqlens_q is None or cu_seqlens_k is None
            or cu_seqlens_q.ndim != 1 or cu_seqlens_k.shape != cu_seqlens_q.shape
            or cu_seqlens_q.numel() < 2
            or cu_seqlens_q.dtype != torch.int32 or cu_seqlens_k.dtype != torch.int32
            or cu_seqlens_q.device != q.device or cu_seqlens_k.device != q.device):
        raise ValueError("Triton ViT attention requires matching GPU int32 packed offsets")
    if not isinstance(max_seqlen_q, int) or max_seqlen_q <= 0:
        raise ValueError("Triton ViT attention requires a positive host max_seqlen_q")

    result = flash_attn_varlen_triton(
        q, k, v, cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale, causal=False, return_softmax_lse=return_lse,
    )
    if return_lse:
        output, lse = result
        return output, lse.transpose(0, 1)
    return result, None
