# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Fixed reduction tiles for the absorbed GLM MLA path on HIP."""

from functools import wraps

import torch
import triton
import triton.language as tl

from . import registry


@triton.jit
def _bmm_fp32(
    lhs, rhs, output, m, n, k, ab, am, ak, bb, bk, bn,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    tile, batch = tl.program_id(0), tl.program_id(1)
    n_tiles = tl.cdiv(n, BN)
    row = (tile // n_tiles) * BM + tl.arange(0, BM)
    col = (tile % n_tiles) * BN + tl.arange(0, BN)
    red = tl.arange(0, BK)
    a = lhs + batch.to(tl.int64) * ab + row[:, None] * am + red[None, :] * ak
    b = rhs + batch.to(tl.int64) * bb + red[:, None] * bk + col[None, :] * bn
    acc = tl.zeros((BM, BN), tl.float32)
    for block in range(tl.cdiv(k, BK)):
        valid_k = red + block * BK < k
        left = tl.load(a + block * BK * ak, (row[:, None] < m) & valid_k[None, :], 0)
        right = tl.load(b + block * BK * bk, valid_k[:, None] & (col[None, :] < n), 0)
        acc = tl.dot(left, right, acc, input_precision="ieee")
    dst = output + batch.to(tl.int64) * m * n + row[:, None] * n + col[None, :]
    tl.store(dst, acc, (row[:, None] < m) & (col[None, :] < n))


def fp32_bmm(lhs, rhs):
    """FP32 scores and value contraction without batch-dependent splitting."""
    batch, m, k = lhs.shape
    n = rhs.shape[-1]
    output = lhs.new_empty((batch, m, n))
    _bmm_fp32[(triton.cdiv(m, 16) * triton.cdiv(n, 32), batch)](
        lhs, rhs, output, m, n, k, *lhs.stride(), *rhs.stride(),
        BM=16, BN=32, BK=16, num_warps=4, num_stages=1,
    )
    return output


def install():
    if torch.version.hip is None:
        return

    from sgl_kernel import flash_attn
    from torch.overrides import TorchFunctionMode
    from sglang.srt.batch_invariant_ops.batch_invariant_ops import bmm_kernel_persistent
    from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA
    from sglang.srt.utils.common import get_device_core_count

    original_attend = flash_attn._attend_chunked
    if getattr(original_attend, "_hcu_fixed_dsa_attention", False):
        return

    class AttentionBmm(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            if func is torch.bmm and len(args) == 2 and not kwargs:
                return fp32_bmm(*args)
            return func(*args, **(kwargs or {}))

    @wraps(original_attend)
    def attend(
        qp, kv_src, seqlens, q_start, scale, causal, window_size,
        softcap, k_descale, v_descale, out_dtype, total_tokens, qvp=None,
    ):
        args = (
            qp, kv_src, seqlens, q_start, scale, causal, window_size, softcap,
            k_descale, v_descale, out_dtype, total_tokens, qvp,
        )
        if (
            qvp is None or not qp.is_cuda or qp.shape[-1] != 1
            or qvp.shape[-1] != 512 or out_dtype != torch.bfloat16
        ):
            return original_attend(*args)
        with AttentionBmm():
            return original_attend(*args)

    def fixed_bf16_bmm(lhs, rhs):
        batch, m, k = lhs.shape
        n = rhs.shape[-1]
        result = lhs.new_empty((batch, m, n))
        cores = get_device_core_count()
        tiles = batch * triton.cdiv(m, 32) * triton.cdiv(n, 128)
        bmm_kernel_persistent[(min(cores, tiles),)](
            lhs, rhs, result, batch, m, n, k, *lhs.stride(), *rhs.stride(),
            *result.stride(), NUM_SMS=cores, A_LARGE=lhs.numel() > 2**31,
            B_LARGE=rhs.numel() > 2**31, C_LARGE=result.numel() > 2**31,
            BLOCK_SIZE_M=32, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32,
            GROUP_SIZE_M=8, num_warps=4, num_stages=1,
        )
        return result

    class AbsorptionBmm(TorchFunctionMode):
        def __init__(self, weight):
            self.weight = weight

        def __torch_function__(self, func, types, args=(), kwargs=None):
            options = kwargs or {}
            if (
                func is not torch.bmm or len(args) != 2 or args[1] is not self.weight
                or args[0].dtype != torch.bfloat16 or not args[0].is_cuda
                or args[0].numel() == 0
            ):
                return func(*args, **options)
            output = fixed_bf16_bmm(*args)
            target = options.get("out")
            if target is not None:
                target.copy_(output)
                return target
            return output

    def patch(name, weight_name):
        original = getattr(DeepseekV2AttentionMLA, name)
        if getattr(original, "_hcu_fixed_absorption", False):
            return

        @wraps(original)
        def forward(self, *args, **kwargs):
            weight = getattr(self, weight_name, None)
            if (
                not self.use_dsa or self.qk_rope_head_dim != 0
                or weight is None or weight.dtype != torch.bfloat16
            ):
                return original(self, *args, **kwargs)
            with AbsorptionBmm(weight):
                return original(self, *args, **kwargs)

        forward._hcu_fixed_absorption = True
        registry.register(
            "glm_mla_matmul", forward,
            target=f"{DeepseekV2AttentionMLA.__module__}:DeepseekV2AttentionMLA.{name}",
            mechanism=registry.MECH_SEAM,
            detail="Use the fixed BF16 tile for the unquantized absorbed MLA projection",
        )
        setattr(DeepseekV2AttentionMLA, name, forward)

    patch("forward_absorb_prepare", "w_kc")
    patch("forward_absorb_core", "w_vc")
    attend._hcu_fixed_dsa_attention = True
    registry.register(
        "glm_mla_matmul", attend,
        target=f"{flash_attn.__name__}:_attend_chunked",
        mechanism=registry.MECH_SEAM,
        detail="Use IEEE FP32 tiles for zero-RoPE MLA scoring and value contraction",
    )
    flash_attn._attend_chunked = attend
