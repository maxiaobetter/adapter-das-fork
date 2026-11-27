# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
import gc
import sys
import types
import pytest
from hcu_adapter import hook_aliases as fast
from sglang.srt.plugins import hook_registry as stock


@pytest.fixture
def native_scan():
    scan = fast._native_scan()
    if scan is None:
        pytest.skip("CPU-native alias helper requires Linux CPython, headers and compiler")
    return scan


def pair(monkeypatch):
    def original(x=1):
        return x + 1
    def wrapped(x=1):
        return x + 10
    source = types.ModuleType("_opt_alias_source")
    source.fn = original
    consumer = types.ModuleType("_opt_alias_consumer")
    consumer.renamed = original
    consumer.second = original
    monkeypatch.setitem(sys.modules, source.__name__, source)
    monkeypatch.setitem(sys.modules, consumer.__name__, consumer)
    return original, wrapped, source, consumer


def test_renamed_aliases_match_stock_and_source_is_excluded(monkeypatch, native_scan):
    original, wrapped, source, consumer = pair(monkeypatch)
    got = fast.propagate(original, wrapped, source, stock._propagate_patch)
    assert got == 2 and source.fn is original
    assert consumer.renamed is wrapped and consumer.second is wrapped
    consumer.renamed = consumer.second = original
    assert stock._propagate_patch(original, wrapped, source) == got


def test_new_module_and_new_alias_are_seen_on_each_call(monkeypatch, native_scan):
    original, wrapped, source, consumer = pair(monkeypatch)
    fast.propagate(original, wrapped, source, stock._propagate_patch)
    added = types.ModuleType("_opt_alias_added")
    added.later = wrapped
    consumer.later = wrapped
    monkeypatch.setitem(sys.modules, added.__name__, added)
    final = lambda x=1: x
    assert fast.propagate(wrapped, final, source, stock._propagate_patch) == 4
    assert added.later is final and consumer.later is final


def test_custom_setter_introducing_alias_into_later_module(monkeypatch, native_scan):
    original, wrapped, source, _ = pair(monkeypatch)
    later = types.ModuleType("_opt_later")
    class Setter(types.ModuleType):
        def __setattr__(self, name, value):
            if name == "held" and value is wrapped:
                later.new_alias = original
            return types.ModuleType.__setattr__(self, name, value)
    custom = Setter("_opt_setter")
    custom.held = original
    monkeypatch.setitem(sys.modules, custom.__name__, custom)
    monkeypatch.setitem(sys.modules, later.__name__, later)
    assert fast.propagate(original, wrapped, source, stock._propagate_patch) == 4
    assert later.new_alias is wrapped


def test_custom_getter_changes_later_module_aliases(monkeypatch, native_scan):
    original, wrapped, source, _ = pair(monkeypatch)
    later = types.ModuleType("_opt_getter_later")
    class Getter(types.ModuleType):
        def __getattribute__(self, name):
            if name == "__dict__":
                later.alias = original
            return super().__getattribute__(name)
    monkeypatch.setitem(sys.modules, "_opt_getter", Getter("_opt_getter"))
    monkeypatch.setitem(sys.modules, later.__name__, later)
    assert fast.propagate(original, wrapped, source, stock._propagate_patch) == 3
    assert later.alias is wrapped


def test_native_unavailable_retains_full_scan(monkeypatch):
    original, wrapped, source, consumer = pair(monkeypatch)
    monkeypatch.setattr(fast, "_native_scan", lambda: None)
    assert fast.propagate(original, wrapped, source, stock._propagate_patch) == 2
    assert consumer.renamed is wrapped


def test_untracked_object_falls_back_without_skipping(monkeypatch):
    assert not gc.is_tracked(int)
    calls = []
    assert fast.propagate(int, str, None, lambda *a: calls.append(a) or 7) == 7
    assert calls == [(int, str, None)]


def test_setter_errors_match_native_propagation(monkeypatch, native_scan):
    original, wrapped, source, _ = pair(monkeypatch)
    class Reject(types.ModuleType):
        def __setattr__(self, name, value):
            if name == "held" and value is wrapped:
                raise ValueError("not permitted")
            return super().__setattr__(name, value)
    custom = Reject("_opt_reject")
    custom.held = original
    monkeypatch.setitem(sys.modules, custom.__name__, custom)
    with pytest.raises(ValueError, match="not permitted"):
        fast.propagate(original, wrapped, source, stock._propagate_patch)


def test_official_registry_cached_apply_class_order_and_wrappers(monkeypatch, native_scan):
    stock.HookRegistry.reset()
    original, _, source, consumer = pair(monkeypatch)
    class Old:
        def method(self, x):
            return x + 1
        @classmethod
        def cm(cls, x):
            return x + 2
        @staticmethod
        def sm(x):
            return x + 3
    class New(Old):
        def method(self, x):
            return x + 4
    source.Target = Old
    consumer.Target = Old
    cached_apply = stock.HookRegistry._apply_target
    native = stock._propagate_patch
    monkeypatch.setenv("HCU_HOOK_ALIAS_FAST", "1")
    assert fast.install()
    patched = stock._propagate_patch
    assert getattr(patched, "_hcu_native_aliases")
    stock.HookRegistry.register(source.__name__+".Target", New, stock.HookType.REPLACE)
    stock.HookRegistry.register(source.__name__+".Target.method", lambda fn, self, x: fn(self, x)*2, stock.HookType.AROUND)
    stock.HookRegistry.register(source.__name__+".Target.cm", lambda result, *a, **k: result+10, stock.HookType.AFTER)
    stock.HookRegistry.register(source.__name__+".Target.sm", lambda result, *a, **k: result+20, stock.HookType.AFTER)
    stock.HookRegistry.register(source.__name__+".fn", lambda fn, x=1: fn(x)*3, stock.HookType.AROUND)
    try:
        target = source.__name__+".fn"
        cached_apply(target, stock.HookRegistry._hooks[target])
        stock.HookRegistry._patched.add(target)
        stock.HookRegistry.apply_hooks()
        assert consumer.Target is New
        assert consumer.Target().method(2) == 12
        assert consumer.Target.cm(2) == 14
        assert consumer.Target.sm(2) == 25
        assert consumer.renamed(2) == 9
        assert consumer.renamed is source.fn
        assert cached_apply.__func__ is stock.HookRegistry._apply_target.__func__
        assert fast.install()
    finally:
        stock._propagate_patch = native
        stock.HookRegistry.reset()


def test_source_drift_keeps_stock(monkeypatch):
    monkeypatch.setenv("HCU_HOOK_ALIAS_FAST", "1")
    def changed(original, wrapped, source_module):
        return -1
    monkeypatch.setattr(stock, "_propagate_patch", changed)
    assert not fast.install()
    assert stock._propagate_patch is changed


def test_disabled_does_not_change_registry(monkeypatch):
    monkeypatch.setenv("HCU_HOOK_ALIAS_FAST", "0")
    original = stock._propagate_patch
    assert not fast.install()
    assert stock._propagate_patch is original


def test_native_cache_integrity_and_missing_compiler(monkeypatch, tmp_path, native_scan):
    from pathlib import Path
    from hcu_adapter import hook_aliases_cpp
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    scan = hook_aliases_cpp.load()
    assert callable(scan)
    manifests = list(tmp_path.rglob("manifest.json"))
    assert len(manifests) == 1
    manifests[0].write_text('{"key":"bad","sha256":"bad"}')
    with pytest.raises(hook_aliases_cpp.NativeCacheError, match="hash mismatch"):
        hook_aliases_cpp.load()
    monkeypatch.setattr(hook_aliases_cpp.shutil, "which", lambda _: None)
    assert hook_aliases_cpp.load() is None


def test_custom_namespace_items_and_setter_protocol(monkeypatch, native_scan):
    original, wrapped, source, _ = pair(monkeypatch)
    calls = []
    class Namespace(dict):
        def items(self):
            calls.append("items")
            return super().items()
    class Custom(types.ModuleType):
        @property
        def __dict__(self):
            return self.namespace
        def __setattr__(self, name, value):
            if name == "held":
                self.namespace[name] = value
            return super().__setattr__(name, value)
    module = Custom("_opt_namespace")
    module.namespace = Namespace()
    module.held = original
    monkeypatch.setitem(sys.modules, "_opt_namespace", module)
    assert fast.propagate(original, wrapped, source, stock._propagate_patch) == 3
    assert module.held is wrapped
    assert calls == ["items"]

def test_new_module_created_during_propagation_matches_snapshot_semantics(monkeypatch, native_scan):
    original, wrapped, source, _ = pair(monkeypatch)
    created = []
    class Creator(types.ModuleType):
        def __setattr__(self, name, value):
            if name == "held" and value is wrapped:
                new = types.ModuleType("_opt_created_late")
                new.alias = original
                created.append(new)
                monkeypatch.setitem(sys.modules, new.__name__, new)
            return super().__setattr__(name, value)
    mod = Creator("_opt_creator")
    mod.held = original
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    assert fast.propagate(original, wrapped, source, stock._propagate_patch) == 3
    assert created[0].alias is original
    assert fast.propagate(original, wrapped, source, stock._propagate_patch) == 1
    assert created[0].alias is wrapped


def test_attribute_and_type_errors_are_skipped_like_stock(monkeypatch, native_scan):
    original, wrapped, source, _ = pair(monkeypatch)
    class Reject(types.ModuleType):
        def __setattr__(self, name, value):
            if name == "held" and value is wrapped:
                raise AttributeError("read-only")
            return super().__setattr__(name, value)
    mod = Reject("_opt_read_only")
    mod.held = original
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    assert fast.propagate(original, wrapped, source, stock._propagate_patch) == 2
    assert mod.held is original


def test_vars_type_error_is_skipped_like_stock(monkeypatch, native_scan):
    original, wrapped, source, _ = pair(monkeypatch)
    class NoDictionary(types.ModuleType):
        @property
        def __dict__(self):
            raise TypeError("no namespace")
    monkeypatch.setitem(sys.modules, "_opt_no_namespace", NoDictionary("_opt_no_namespace"))
    assert fast.propagate(original, wrapped, source, stock._propagate_patch) == 2
