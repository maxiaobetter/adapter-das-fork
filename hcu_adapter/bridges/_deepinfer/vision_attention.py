# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Explicit provider for the FlashAttention entry used by upstream ViT."""

from hcu_adapter import registry
from hcu_adapter.ops import selector


def install():
    import flash_attn.cute.interface as interface

    op = "flash_attn.cute.interface._flash_attn_fwd"
    provider = selector.provider(op, "torch")
    current = getattr(interface, "_flash_attn_fwd", None)
    if current is not None and current is not provider and not getattr(current, "_hcu_unimplemented", False):
        raise RuntimeError("An existing FlashAttention ViT provider is already installed")
    implementation = f"{provider.__module__}.{provider.__name__}"
    registry.register(op, implementation, op, "rebind")
    interface._flash_attn_fwd = provider
    print(f"HCU adapter runtime: ViT varlen attention -> {implementation}", flush=True)
