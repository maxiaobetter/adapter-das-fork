# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from functools import partial, wraps

from . import registry


def install_ordered_tp_reduce():
    import torch
    from sglang.srt.distributed import parallel_state

    from .ordered_tp_sum import sum_bf16_tp8

    coordinator = parallel_state.GroupCoordinator
    original = coordinator._all_reduce_in_place
    if torch.version.hip is None or getattr(original, "_hcu_ordered_tp_reduce", False):
        return

    @wraps(original)
    def ordered_tp_reduce(self, input_):
        if (
            not input_.is_cuda
            or input_.dtype != torch.bfloat16
            or self.world_size != 8
            or self not in (parallel_state.get_tp_group(), parallel_state.get_attn_tp_group())
        ):
            return original(self, input_)
        if input_.numel() == 0:
            return

        flat_input = input_.contiguous().view(-1)
        parts = torch.empty(
            (self.world_size, flat_input.numel()),
            dtype=input_.dtype,
            device=input_.device,
        )
        self._all_gather_into_tensor(parts.view(-1), flat_input)
        sum_bf16_tp8(parts, flat_input)
        if not input_.is_contiguous():
            input_.copy_(flat_input.view(input_.shape))

    registry.register(
        "allreduce", ordered_tp_reduce,
        target=f"{parallel_state.__name__}:GroupCoordinator._all_reduce_in_place",
        mechanism=registry.MECH_SEAM,
        detail="8-rank BF16 TP all-gather and fixed rank order sum",
    )
    ordered_tp_reduce._hcu_ordered_tp_reduce = True
    coordinator._all_reduce_in_place = ordered_tp_reduce


def install():
    import torch
    from sglang.srt.distributed.device_communicators import custom_all_reduce as module

    if torch.version.hip is None or not module._is_cuda:
        return
    if getattr(module.dispatch_custom_allreduce, "_hcu_aiter_dispatch", False):
        return

    def dispatch_custom_allreduce(group, device):
        if not module.get_bool_env_var("SGLANG_USE_AITER_AR", default="true"):
            raise RuntimeError("HCU AITER all-reduce is disabled")
        if (
            module.envs.SGLANG_USE_1STAGE_ALLREDUCE.get()
            or module.envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.get()
        ):
            raise RuntimeError("HCU AITER all-reduce does not implement the requested deterministic path")
        from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce

        return partial(
            CustomAllreduce,
            enable_register_for_capturing=not module.envs.SGLANG_MEMORY_SAVER_CUDA_GRAPH.get(),
        )

    registry.register(
        "allreduce", "aiter_dist",
        target=module.__name__ + ".dispatch_custom_allreduce",
        mechanism=registry.MECH_REBIND,
        detail="use the official HIP provider when SGLang uses CUDA platform semantics",
    )
    dispatch_custom_allreduce._hcu_aiter_dispatch = True
    module.dispatch_custom_allreduce = dispatch_custom_allreduce
