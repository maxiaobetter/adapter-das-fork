# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Batch-invariant FP32 pre-normalization statistics for GLM mHC on HIP."""

from functools import wraps

import torch

from . import registry


def install():
    if torch.version.hip is None:
        return

    from sglang.kernels.ops.layernorm import mhc
    from sglang.srt.models import glm5_next

    original = glm5_next._hc_pre_fn
    if getattr(original, "_hcu_invariant_stats", False):
        return

    @wraps(original)
    @torch._dynamo.disable
    def hc_pre(
        x, hc_fn, hc_scale, hc_base, hc_mult, rms_eps, hc_eps, sinkhorn_iters,
        post_mult_value=2.0, hc_norm_weight=None, out_norm_weight=None,
        out_norm_eps=None,
    ):
        if (
            mhc._use_aiter_mhc()
            or mhc._use_tilelang_mhc_pre()
            or not mhc._is_hip
            or mhc._HC_MIX_DOT_PRECISION != "ieee"
            or not x.is_cuda
            or x.dtype != torch.bfloat16
            or hc_fn.dtype != torch.float32
            or hc_norm_weight is not None
            or x.ndim != 2
            or x.numel() == 0
            or x.stride(1) != 1
            or hc_fn.stride(1) != 1
            or x.shape[1] % mhc._HC_MIX_BLOCK_K
        ):
            return original(
                x, hc_fn, hc_scale, hc_base, hc_mult, rms_eps, hc_eps,
                sinkhorn_iters, post_mult_value, hc_norm_weight,
                out_norm_weight, out_norm_eps,
            )

        rows, width = x.shape
        hc = hc_mult
        residual = x.view(rows, hc, width // hc)
        # Keep the unfused Torch mixing, Sinkhorn and output casts. Only the
        # FP32 dot-product and RMS statistics switch to fixed upstream tiles.
        mixes = mhc.hc_mix_stats(x, hc_fn, rms_eps)
        pre = torch.sigmoid(mixes[:, :hc] * hc_scale[0] + hc_base[:hc]) + hc_eps
        post = post_mult_value * torch.sigmoid(
            mixes[:, hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc]
        )
        comb = mixes[:, 2 * hc :].view(rows, hc, hc)
        comb = comb * hc_scale[2] + hc_base[2 * hc :].view(hc, hc)
        comb = comb.softmax(-1) + hc_eps
        comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
        for _ in range(sinkhorn_iters - 1):
            comb = comb / (comb.sum(-1, keepdim=True) + hc_eps)
            comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
        hidden = (pre.unsqueeze(-1) * residual.float()).sum(dim=1).to(x.dtype)
        return hidden, comb.flatten(1), post, False

    hc_pre._hcu_invariant_stats = True
    registry.register(
        "glm_mhc_stats", hc_pre,
        target=f"{glm5_next.__name__}:_hc_pre_fn",
        mechanism=registry.MECH_SEAM,
        detail="Use fixed FP32 mix statistics for GLM's unfused mHC boundary",
    )
    glm5_next._hc_pre_fn = hc_pre
