# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import os
from importlib.metadata import entry_points


def plugin_owns_slimquant():
    allowed = os.environ.get("SGLANG_PLUGINS", "")
    if allowed and "hcu" not in {name.strip() for name in allowed.split(",")}:
        return False
    selected_platform = os.environ.get("SGLANG_PLATFORM", "")
    excluded = set()
    if selected_platform:
        excluded = {
            ep.dist.name
            for ep in entry_points(group="sglang.srt.platforms")
            if ep.dist and ep.name != selected_platform
        }
    return any(
        ep.name == "hcu" and ep.value == "hcu_sglang:register"
        and (not ep.dist or ep.dist.name not in excluded)
        for ep in entry_points(group="sglang.srt.plugins")
    )


def activate_legacy_slimquant():
    if plugin_owns_slimquant():
        return False
    from hcu_adapter import slimquant

    slimquant.activate()
    return True
