# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in acceleration of the existing SGLang Hook alias propagation contract."""

import functools
import gc
import hashlib
import inspect
import logging
import subprocess
import sys

LOG = logging.getLogger("hcu.adapter")
EXPECTED_SOURCE_SHA256 = "49c03f76266054fb81be18c81c89419c60716d79bf7857729c2aff0e762d5e61"


_NATIVE = None
_NATIVE_READY = False


def _native_scan():
    global _NATIVE, _NATIVE_READY
    if not _NATIVE_READY:
        from hcu_adapter.hook_aliases_cpp import load, NativeCacheError
        try:
            _NATIVE = load()
        except NativeCacheError:
            raise
        except (OSError, ImportError, RuntimeError, subprocess.SubprocessError):
            _NATIVE = None
        _NATIVE_READY = True
    return _NATIVE


def _scan_namespace(mod, namespace, original, wrapped):
    patched = 0
    for name, value in list(namespace.items()):
        if value is original:
            try:
                setattr(mod, name, wrapped)
                patched += 1
            except (AttributeError, TypeError):
                pass
    return patched


def propagate(original, wrapped, source_module, fallback):
    """Scan a fresh module snapshot; keep the official path for unsupported cases."""
    if (
        sys.implementation.name != "cpython"
        or not gc.is_tracked(original)
        or wrapped is original
    ):
        return fallback(original, wrapped, source_module)
    scan = _native_scan()
    if scan is None:
        return fallback(original, wrapped, source_module)
    return scan(
        original, wrapped, source_module, _scan_namespace, list(sys.modules.values())
    )


def install():
    """Bind only the fingerprinted upstream helper, leaving official files intact."""
    from hcu_adapter.infra import hcu_env, registry

    if not hcu_env.enabled("HCU_HOOK_ALIAS_FAST"):
        return False
    from sglang.srt.plugins import hook_registry

    original = hook_registry._propagate_patch
    if getattr(original, "_hcu_native_aliases", False):
        return True
    if tuple(inspect.signature(original).parameters) != ("original", "wrapped", "source_module"):
        LOG.warning("HCU hook alias optimization: unsupported signature; retaining stock")
        return False
    try:
        digest = hashlib.sha256(inspect.getsource(original).encode()).hexdigest()
    except (OSError, TypeError):
        return False
    if digest != EXPECTED_SOURCE_SHA256:
        LOG.warning("HCU hook alias optimization: source changed; retaining stock")
        return False

    if _native_scan() is None:
        LOG.warning("HCU hook alias optimization: native helper unavailable; retaining stock")
        return False

    @functools.wraps(original)
    def replacement(original, wrapped, source_module):
        return propagate(original, wrapped, source_module, replacement.__wrapped__)

    replacement._hcu_native_aliases = True
    hook_registry._propagate_patch = replacement
    registry.register(
        "hook_alias_propagation", "native-reference-scan",
        target="sglang.srt.plugins.hook_registry._propagate_patch",
        mechanism=registry.MECH_REBIND,
        detail="CPython identity scan; stock fallback when unsupported",
    )
    return True
