# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Batch-invariant BF16 projections for the fused GLM-5.3 attention path."""

from functools import wraps

import torch
import triton

from . import registry


# The upstream default tile exceeds the LDS capacity on gfx938.
_TILE = {
    "BLOCK_SIZE_M": 32,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 32,
    "GROUP_SIZE_M": 8,
    "num_stages": 1,
    "num_warps": 4,
}


def install():
    if torch.version.hip is None:
        return

    from sglang.srt.batch_invariant_ops.batch_invariant_ops import (
        bmm_kernel_persistent,
        matmul_kernel_persistent,
    )
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
    from sglang.srt.models.glm5_next import Glm5NextLinearAttention
    from sglang.srt.utils.common import get_device_core_count

    original_init = Glm5NextLinearAttention.__init__
    if getattr(original_init, "_hcu_invariant_projection", False):
        return

    def use_fixed_tile(x, weight, dims):
        return (
            x.ndim == dims
            and x.is_cuda
            and x.dtype == weight.dtype == torch.bfloat16
            and x.numel() > 0
        )

    def fix_linear_projection(module):
        method = module.quant_method
        if type(method) is not UnquantizedLinearMethod:
            return
        original_apply = method.apply

        @wraps(original_apply)
        def apply(layer, x, bias=None):
            weight = layer.weight
            if bias is not None or not use_fixed_tile(x, weight, 2):
                return original_apply(layer, x, bias)

            rows, k = x.shape
            b = weight.T
            n = b.shape[1]
            output = x.new_empty((rows, n))
            n_sms = get_device_core_count()
            tiles = triton.cdiv(rows, _TILE["BLOCK_SIZE_M"]) * triton.cdiv(
                n, _TILE["BLOCK_SIZE_N"]
            )
            matmul_kernel_persistent[(min(n_sms, tiles),)](
                x, b, output, None, rows, n, k,
                *x.stride(), *b.stride(), *output.stride(),
                NUM_SMS=n_sms,
                A_LARGE=x.numel() > 2**31,
                B_LARGE=weight.numel() > 2**31,
                C_LARGE=output.numel() > 2**31,
                HAS_BIAS=False,
                **_TILE,
            )
            return output

        method.apply = apply

    def fix_gate_projection(module):
        original_forward = module.forward

        @wraps(original_forward)
        def forward(input):
            if not use_fixed_tile(input, module.weight, 3):
                return original_forward(input)

            weight = module.weight.transpose(-1, -2)
            batch, rows, k = input.shape
            n = weight.shape[-1]
            output = input.new_empty((batch, rows, n))
            n_sms = get_device_core_count()
            tiles = (
                batch
                * triton.cdiv(rows, _TILE["BLOCK_SIZE_M"])
                * triton.cdiv(n, _TILE["BLOCK_SIZE_N"])
            )
            bmm_kernel_persistent[(min(n_sms, tiles),)](
                input, weight, output, batch, rows, n, k,
                *input.stride(), *weight.stride(), *output.stride(),
                NUM_SMS=n_sms,
                A_LARGE=input.numel() > 2**31,
                B_LARGE=weight.numel() > 2**31,
                C_LARGE=output.numel() > 2**31,
                **_TILE,
            )
            return output

        module.forward = forward

    @wraps(original_init)
    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if self.do_fuse_qkvbfg:
            fix_linear_projection(self.fused_qkvbfg_a_proj)
            fix_gate_projection(self.fused_fg_b_proj)
            fix_linear_projection(self.o_proj)

    registry.register(
        "kda_projection", init,
        target=f"{Glm5NextLinearAttention.__module__}:Glm5NextLinearAttention.__init__",
        mechanism=registry.MECH_SEAM,
        detail="Use a fixed GEMM tile for unquantized BF16 GLM-5.3 KDA projections",
    )
    init._hcu_invariant_projection = True
    Glm5NextLinearAttention.__init__ = init
