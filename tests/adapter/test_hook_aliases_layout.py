# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""The Hook alias optimizer must load through main's hcu_adapter.infra layout.

Release PR #206 wrote ``hook_aliases`` against the pre-refactor layout
(``from hcu_adapter import hcu_env, registry``). The sglang bridge calls
``hook_aliases.install()`` unguarded between the layer-dump and deep_ep
sections, so an ImportError there aborts every bridge installed after it --
with the opt-in switch off, i.e. on every launch. No sglang import is
needed: with the switch off, install() returns before touching sglang, so
this case runs on every host.
"""

import pytest

from hcu_adapter import hook_aliases


@pytest.mark.parametrize("switch", [None, "0"], ids=["unset", "0"])
def test_install_with_the_switch_off_loads_through_the_infra_layout(monkeypatch, switch):
    monkeypatch.delenv("HCU_HOOK_ALIAS_FAST", raising=False)
    if switch is not None:
        monkeypatch.setenv("HCU_HOOK_ALIAS_FAST", switch)
    assert hook_aliases.install() is False
