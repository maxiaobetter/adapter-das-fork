# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "provider_under_test",
    ROOT / "hcu_adapter/slimquant_provider.py",
)
provider = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(provider)


class ProviderOwnershipTests(unittest.TestCase):
    def test_selection(self):
        hcu = SimpleNamespace(name="hcu", value="hcu_sglang:register", dist=None)
        other = SimpleNamespace(name="hcu", value="another_plugin:register", dist=None)
        cases = [
            ({}, [], False),
            ({}, [hcu], True),
            ({"SGLANG_PLUGINS": ""}, [hcu], True),
            ({"SGLANG_PLUGINS": "other, hcu "}, [hcu], True),
            ({"SGLANG_PLUGINS": "other"}, [hcu], False),
            ({"SGLANG_PLUGINS": "hcu"}, [], False),
            ({"SGLANG_PLUGINS": "hcu"}, [other], False),
        ]
        for env, entries, expected in cases:
            with self.subTest(env=env, entries=entries):
                with patch.dict(os.environ, env, clear=True), patch.object(
                    provider, "entry_points", return_value=entries
                ):
                    self.assertEqual(provider.plugin_owns_slimquant(), expected)

    def test_unselected_platform_distribution_is_excluded(self):
        dist = SimpleNamespace(name="hcu-sglang-plugin")
        hcu = SimpleNamespace(name="hcu", value="hcu_sglang:register", dist=dist)
        platform = SimpleNamespace(name="hcu-platform", dist=dist)
        def discover(*, group):
            return [platform] if group == "sglang.srt.platforms" else [hcu]
        with patch.dict(os.environ, {"SGLANG_PLATFORM": "other"}, clear=True), patch.object(
            provider, "entry_points", side_effect=discover
        ):
            self.assertFalse(provider.plugin_owns_slimquant())

    def test_plugin_prevents_all_legacy_activation(self):
        legacy = Mock()
        with patch.object(provider, "plugin_owns_slimquant", return_value=True), patch.dict(
            sys.modules, {"hcu_adapter": SimpleNamespace(slimquant=legacy)}
        ):
            self.assertFalse(provider.activate_legacy_slimquant())
        legacy.activate.assert_not_called()

    def test_adapter_only_retains_legacy_activation(self):
        legacy = Mock()
        with patch.object(provider, "plugin_owns_slimquant", return_value=False), patch.dict(
            sys.modules, {"hcu_adapter": SimpleNamespace(slimquant=legacy)}
        ):
            self.assertTrue(provider.activate_legacy_slimquant())
        legacy.activate.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
