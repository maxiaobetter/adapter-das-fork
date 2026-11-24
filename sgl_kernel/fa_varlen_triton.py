# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""SGLang's existing import path for the shared packed attention kernel."""

from hcu_adapter.kernels.fa_varlen_triton import flash_attn_varlen_triton

__all__ = ["flash_attn_varlen_triton"]
