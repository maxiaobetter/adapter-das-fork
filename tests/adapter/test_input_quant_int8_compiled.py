# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Portable control contracts; Torch/vendor doubles execute no device math."""

from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[2] / (
    "hcu_adapter/bridges/vllm_input_quant_int8_lightop.py"
)


class Tensor:
    def __init__(self, shape=(33, 1536), dtype="bf16", contiguous=True,
                 cuda=True, pointer=16, device_index=0):
        self.shape = shape
        self.dtype = dtype
        self.device = SimpleNamespace(index=device_index)
        self.is_cuda = cuda
        self._contiguous = contiguous
        self.pointer = pointer
        self.value = None

    def dim(self):
        return len(self.shape)

    def is_contiguous(self):
        return self._contiguous

    def numel(self):
        return math.prod(self.shape)

    def data_ptr(self):
        if STATE.compiling:
            raise AssertionError("Dynamo must not inspect a device pointer")
        return self.pointer

    def contiguous(self):
        return self


STATE = SimpleNamespace()


class CustomOp:
    def __init__(self, function):
        self.real = function
        self.fake = None
        self.calls = []

    def register_fake(self, function):
        self.fake = function

    def __call__(self, *args):
        self.calls.append(args)
        return (self.fake if STATE.compiling else self.real)(*args)


def load_bridge():
    torch = ModuleType("torch")
    torch.Tensor = Tensor
    torch.bfloat16 = "bf16"
    torch.float32 = "fp32"
    torch.int8 = "int8"
    torch.compiler = SimpleNamespace(is_compiling=lambda: STATE.compiling)

    def empty_like(tensor, dtype):
        result = Tensor(tensor.shape, dtype)
        STATE.allocations.append(result)
        return result

    def empty(shape, *, dtype, device):
        result = Tensor(tuple(shape), dtype)
        result.device = device
        STATE.allocations.append(result)
        return result

    def ones(hidden, *, device, dtype):
        return Tensor((hidden,), dtype)

    def properties(device):
        if STATE.compiling or not STATE.executing:
            raise AssertionError("GPU query outside opaque concrete execution")
        STATE.property_queries.append(device)
        if STATE.arch is None:
            return SimpleNamespace()
        return SimpleNamespace(gcnArchName=STATE.arch + ":sramecc+:xnack-")

    def original_native(output, input, scales, azp):
        STATE.native_calls.append((output, input, scales, azp))
        output.value = "core-quantized"
        scales.value = "core-scales"

    def custom_op(name, function, *, mutates_args, schema):
        result = CustomOp(function)
        STATE.publications.append((name, mutates_args, schema, result))
        return result

    torch.empty_like = empty_like
    torch.empty = empty
    torch.ones = ones
    torch.cuda = SimpleNamespace(
        get_device_properties=properties,
        current_device=lambda: 0,
        is_current_stream_capturing=lambda: STATE.capturing,
        current_stream=lambda device: SimpleNamespace(cuda_stream=7),
    )
    torch.ops = SimpleNamespace(_C=SimpleNamespace(
        dynamic_scaled_int8_quant=original_native))
    torch.library = SimpleNamespace(custom_op=custom_op)
    spec = importlib.util.spec_from_file_location("int8_bridge_under_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"torch": torch}):
        spec.loader.exec_module(module)
    module._log_once = lambda *args, **kwargs: None

    def lightop(output, input, scales, smooth, round_after_smooth, scale_floor):
        STATE.lightop_calls.append((output, input, scales, smooth,
                                    round_after_smooth, scale_floor))
        if STATE.native_error:
            raise RuntimeError("real native failure")
        output.value = "lightop-quantized"
        scales.value = "lightop-scales"

    module._OP = lightop
    return module


class CompiledQuantTests(unittest.TestCase):
    def setUp(self):
        STATE.__dict__.clear()
        STATE.__dict__.update(
            compiling=True, executing=False, arch="gfx936", capturing=False,
            allocations=[], publications=[], property_queries=[],
            native_calls=[], lightop_calls=[], original_calls=[],
            native_error=False,
        )
        self.environment = patch.dict(os.environ, {
            "HCU_LIGHTOP_PER_TOKEN_INT8_GRAPH": "1",
        })
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.bridge = load_bridge()

    def original(self, input, scale=None, azp=None, symmetric=True):
        STATE.original_calls.append((input, scale, azp, symmetric))
        return "original", scale, azp

    def wrap(self):
        return self.bridge._wrap(self.original)

    def compiled(self, input=None):
        input = Tensor() if input is None else input
        result = self.wrap()(input)
        self.assertEqual(len(STATE.publications), 1)
        self.assertFalse(STATE.original_calls)
        self.assertIsNone(result[2])
        self.assertEqual(result[0].shape, input.shape)
        self.assertEqual(result[0].dtype, "int8")
        self.assertEqual(result[1].shape, (input.shape[0], 1))
        self.assertEqual(result[1].dtype, "fp32")
        self.assertFalse(STATE.property_queries)
        self.assertFalse(STATE.lightop_calls)
        self.assertFalse(STATE.native_calls)
        return result, STATE.publications[0][3]

    def execute(self, op):
        STATE.compiling = False
        STATE.executing = True
        return op.real(*op.calls[-1])

    def test_compiled_wrapper_and_fake_preserve_three_result_contract(self):
        result, op = self.compiled()
        self.assertIsNone(op.fake(*op.calls[-1]))
        self.assertEqual(len(STATE.allocations), 2)
        self.assertIs(result[0], op.calls[-1][0])
        self.assertIs(result[1], op.calls[-1][2])

    def test_custom_op_declares_both_output_mutations(self):
        self.compiled()
        name, mutations, schema, _ = STATE.publications[0]
        self.assertEqual(name, "hcu_adapter::vllm_lightop_compiled_int8_quant")
        self.assertEqual(set(mutations), {"output", "scales"})
        self.assertIn("-> ()", schema)

    def test_gfx936_runtime_uses_existing_public_lightop_outputs(self):
        result, op = self.compiled()
        self.execute(op)
        self.assertEqual(len(STATE.lightop_calls), 1)
        self.assertFalse(STATE.native_calls)
        self.assertEqual(len(STATE.allocations), 2)
        self.assertIs(STATE.lightop_calls[0][0], result[0])
        self.assertIs(STATE.lightop_calls[0][2], result[1])
        self.assertEqual(STATE.lightop_calls[0][-2:], (False, -1.0))

    def test_other_architectures_keep_core_operator_and_allocations(self):
        for arch in ("gfx938", "gfx928", "gfx950", None):
            with self.subTest(arch=arch):
                STATE.arch = arch
                result, op = self.compiled()
                self.execute(op)
                self.assertFalse(STATE.lightop_calls)
                self.assertIs(STATE.native_calls[-1][0], result[0])
                self.assertIs(STATE.native_calls[-1][2], result[1])
                self.assertIsNone(STATE.native_calls[-1][3])
                STATE.compiling = True
                STATE.executing = False
                STATE.property_queries.clear()
                STATE.native_calls.clear()

    def test_graph_switch_off_keeps_original_compiled_call(self):
        os.environ["HCU_LIGHTOP_PER_TOKEN_INT8_GRAPH"] = "0"
        self.assertEqual(self.wrap()(Tensor())[0], "original")
        self.assertFalse(STATE.publications)

    def test_unsupported_metadata_and_parameters_keep_original(self):
        calls = [
            (Tensor(), {"scale": object()}),
            (Tensor(), {"azp": object()}),
            (Tensor(), {"symmetric": False}),
            (Tensor(dtype="fp16"), {}),
            (Tensor(contiguous=False), {}),
            (Tensor(cuda=False), {}),
            (Tensor(shape=(2, 3, 1536)), {}),
            (Tensor(shape=(0, 1536)), {}),
            (Tensor(shape=(33, 0)), {}),
            (Tensor(shape=(33, 17)), {}),
            (Tensor(shape=(33, 8208)), {}),
        ]
        wrapped = self.wrap()
        for input, kwargs in calls:
            with self.subTest(input=input.shape, kwargs=kwargs):
                self.assertEqual(wrapped(input, **kwargs)[0], "original")
        self.assertFalse(STATE.property_queries)
        self.assertFalse(STATE.lightop_calls)

    def test_runtime_misalignment_retains_existing_native_fallback(self):
        _, op = self.compiled(Tensor(pointer=17))
        self.execute(op)
        self.assertFalse(STATE.lightop_calls)
        self.assertEqual(len(STATE.native_calls), 1)

    def test_gfx936_native_errors_propagate_without_core_fallback(self):
        for capturing in (False, True):
            with self.subTest(capturing=capturing):
                STATE.capturing = capturing
                STATE.compiling = True
                STATE.executing = False
                STATE.property_queries.clear()
                STATE.lightop_calls.clear()
                _, op = self.compiled()
                STATE.native_error = True
                with self.assertRaisesRegex(RuntimeError, "real native failure"):
                    self.execute(op)
                self.assertFalse(STATE.native_calls)

    def test_eager_existing_lightop_path_is_retained(self):
        wrapped = self.wrap()
        STATE.compiling = False
        STATE.executing = True
        output, scales, azp = wrapped(Tensor())
        self.assertEqual(output.value, "lightop-quantized")
        self.assertEqual(scales.value, "lightop-scales")
        self.assertIsNone(azp)

    def test_install_master_off_leaves_binding_and_gpu_untouched(self):
        infra = ModuleType("hcu_adapter.infra")
        infra.hcu_env = SimpleNamespace(enabled=lambda name: False)
        host = ModuleType("hcu_adapter")
        host.infra = infra
        module = SimpleNamespace(scaled_int8_quant=self.original)
        with patch.dict(sys.modules, {"hcu_adapter": host,
                                     "hcu_adapter.infra": infra}):
            self.assertFalse(self.bridge.install(module))
        self.assertEqual(module.scaled_int8_quant, self.original)
        self.assertFalse(STATE.publications)
        self.assertFalse(STATE.property_queries)


if __name__ == "__main__":
    unittest.main(verbosity=2)
