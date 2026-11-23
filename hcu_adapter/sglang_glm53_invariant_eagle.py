# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Stable EH normalization and BF16 LM-head projection for GLM's EAGLE path."""

from functools import wraps

import torch
import triton

from . import registry


def install():
    if torch.version.hip is None:
        return

    from sglang.srt.batch_invariant_ops.batch_invariant_ops import (
        matmul_kernel_persistent,
        rms_norm_batch_invariant,
    )
    from sglang.srt.layers.logits_processor import should_apply_lm_head_quant_method
    from sglang.srt.models import deepseek_nextn
    from sglang.srt.models.glm5_next import Glm5NextForConditionalGeneration
    from sglang.srt.models.glm5_next_nextn import Glm5NextForConditionalGenerationNextN
    from sglang.srt.utils.common import get_device_core_count

    original_eh_norm = deepseek_nextn.fused_eh_norm
    if getattr(original_eh_norm, "_hcu_invariant_eagle", False):
        return

    def is_bf16_matrix(value):
        return value.is_cuda and value.ndim == 2 and value.dtype == torch.bfloat16

    @wraps(original_eh_norm)
    def eh_norm(inputs_embeds, previous_hidden, enorm_weight, hnorm_weight, eps):
        if (
            not is_bf16_matrix(inputs_embeds)
            or not is_bf16_matrix(previous_hidden)
            or inputs_embeds.numel() == 0
            or previous_hidden.shape != inputs_embeds.shape
            or inputs_embeds.shape[1] != 4096
        ):
            return original_eh_norm(
                inputs_embeds, previous_hidden, enorm_weight, hnorm_weight, eps
            )
        embedded = rms_norm_batch_invariant(inputs_embeds, enorm_weight, eps)
        previous = rms_norm_batch_invariant(previous_hidden, hnorm_weight, eps)
        return torch.cat((embedded, previous), dim=-1)

    def project(x, weight):
        rows, k = x.shape
        n = weight.shape[0]
        b = weight.T
        output = x.new_empty((rows, n))
        cores = get_device_core_count()
        tiles = triton.cdiv(rows, 32) * triton.cdiv(n, 128)
        matmul_kernel_persistent[(min(cores, tiles),)](
            x, b, output, None, rows, n, k, *x.stride(), *b.stride(),
            *output.stride(), NUM_SMS=cores, A_LARGE=x.numel() > 2**31,
            B_LARGE=b.numel() > 2**31, C_LARGE=output.numel() > 2**31,
            HAS_BIAS=False, BLOCK_SIZE_M=32, BLOCK_SIZE_N=128,
            BLOCK_SIZE_K=32, GROUP_SIZE_M=8, num_warps=4, num_stages=1,
        )
        return output

    def install_head(model):
        original_init = model.__init__

        @wraps(original_init)
        def init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            processor = self.logits_processor
            if processor is None:
                return
            original_head = processor._compute_lm_head

            @wraps(original_head)
            def compute(hidden_states, lm_head, embedding_bias=None):
                weight = getattr(lm_head, "weight", None)
                if (
                    embedding_bias is not None
                    or processor.use_fp32_lm_head
                    or hasattr(lm_head, "set_lora")
                    or should_apply_lm_head_quant_method(
                        lm_head, getattr(lm_head, "quant_method", None)
                    )
                    or not is_bf16_matrix(hidden_states)
                    or hidden_states.numel() == 0
                    or weight is None
                    or not is_bf16_matrix(weight)
                ):
                    return original_head(hidden_states, lm_head, embedding_bias)
                return project(hidden_states, weight)

            processor._compute_lm_head = compute

        model.__init__ = init

    eh_norm._hcu_invariant_eagle = True
    registry.register(
        "glm_eagle_projections", eh_norm,
        target=f"{deepseek_nextn.__name__}:fused_eh_norm",
        mechanism=registry.MECH_SEAM,
        detail="Use row-wise RMS normalization and fixed BF16 LM-head tiles on HIP",
    )
    deepseek_nextn.fused_eh_norm = eh_norm
    install_head(Glm5NextForConditionalGeneration)
    install_head(Glm5NextForConditionalGenerationNextN)
