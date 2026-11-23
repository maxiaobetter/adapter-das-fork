# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Fixed-tile dense W8A8 GEMM for the 128x128 E4M3 weight layout."""

import torch
import triton


def gemm_nt(lhs, rhs, out):
    """Write the result in place, returning None for an unsupported layout."""
    a, a_scale = lhs[:2]
    b, b_scale = rhs[:2]
    m, k = a.shape
    n = b.shape[0]
    block = 128
    if (
        torch.version.hip is None
        or not a.is_cuda
        or a.dtype != torch.float8_e4m3fn
        or b.dtype != a.dtype
        or out.dtype != torch.bfloat16
        or b.shape != (n, k)
        or out.shape != (m, n)
        or a_scale.dtype != torch.float32
        or b_scale.dtype != torch.float32
        or not m
        or not n
        or not k
        or n % block
        or k % block
        or a_scale.shape != (m, k // block)
        or b_scale.shape != (n // block, k // block)
    ):
        return None

    # Use the reference Triton kernel without its M-dependent tuner.
    # JIT 唯一实现：使用未经过形状调优分发的 Triton kernel。
    from sglang.kernels.ops.gemm.fp8_kernel import _w8a8_block_fp8_matmul

    block_m, block_n = 32, 64
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
    _w8a8_block_fp8_matmul[grid](
        a, b, out, a_scale, b_scale, m, n, k, block, block,
        *a.stride(), b.stride(1), b.stride(0), *out.stride(),
        *a_scale.stride(), b_scale.stride(1), b_scale.stride(0),
        BLOCK_SIZE_M=block_m, BLOCK_SIZE_N=block_n, BLOCK_SIZE_K=block,
        GROUP_SIZE_M=1, num_warps=4, num_stages=1, needs_masking=False,
    )
    return out
