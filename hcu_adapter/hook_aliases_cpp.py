# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Build and validate the CPU-only CPython alias scanner in a private cache."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import sysconfig
import tempfile

C_SOURCE = """/* Copyright (c) 2026 Hygon Information Technology Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>

static PyObject *scan(PyObject *self, PyObject *args) {
    PyObject *original, *wrapped, *source, *fallback;
    PyObject *modules = NULL, *ns = NULL, *items = NULL;
    Py_ssize_t patched = 0;
    if (!PyArg_ParseTuple(args, "OOOOO", &original, &wrapped, &source, &fallback, &modules))
        return NULL;
    if (!PyList_CheckExact(modules)) {
        PyErr_SetString(PyExc_TypeError, "modules must be a list snapshot");
        return NULL;
    }
    Py_INCREF(modules);
    for (Py_ssize_t mi = 0; mi < PyList_GET_SIZE(modules); mi++) {
        PyObject *mod = PyList_GET_ITEM(modules, mi);
        if (mod == source || !PyModule_Check(mod)) continue;
        PyObject *vars_fn = PyDict_GetItemString(PyEval_GetBuiltins(), "vars");
        if (!vars_fn) {
            PyErr_SetString(PyExc_RuntimeError, "vars builtin is unavailable");
            goto error;
        }
        ns = PyObject_CallOneArg(vars_fn, mod);
        if (!ns) {
            if (PyErr_ExceptionMatches(PyExc_TypeError)) { PyErr_Clear(); continue; }
            goto error;
        }
        if (!PyDict_CheckExact(ns)) {
            PyObject *n = PyObject_CallFunctionObjArgs(fallback, mod, ns, original, wrapped, NULL);
            if (!n) goto error;
            Py_ssize_t count = PyLong_AsSsize_t(n);
            Py_DECREF(n);
            if (count == -1 && PyErr_Occurred()) goto error;
            patched += count;
            Py_CLEAR(ns);
            continue;
        }
        int found = 0;
        Py_ssize_t pos = 0;
        PyObject *key, *value;
        while (PyDict_Next(ns, &pos, &key, &value)) {
            if (value == original) { found = 1; break; }
        }
        if (!found) { Py_CLEAR(ns); continue; }
        items = PyDict_Items(ns);
        if (!items) goto error;
        for (Py_ssize_t ai = 0; ai < PyList_GET_SIZE(items); ai++) {
            PyObject *pair = PyList_GET_ITEM(items, ai);
            key = PyTuple_GET_ITEM(pair, 0);
            value = PyTuple_GET_ITEM(pair, 1);
            if (value != original) continue;
            if (PyObject_SetAttr(mod, key, wrapped) == 0) {
                patched++;
            } else if (PyErr_ExceptionMatches(PyExc_AttributeError) ||
                       PyErr_ExceptionMatches(PyExc_TypeError)) {
                PyErr_Clear();
            } else {
                goto error;
            }
        }
        Py_CLEAR(items);
        Py_CLEAR(ns);
    }
    Py_DECREF(modules);
    return PyLong_FromSsize_t(patched);
error:
    Py_XDECREF(items);
    Py_XDECREF(ns);
    Py_XDECREF(modules);
    return NULL;
}
static PyMethodDef methods[] = {
    {"scan", scan, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}
};
static struct PyModuleDef definition = {
    PyModuleDef_HEAD_INIT, "_hcu_hook_alias_scan", NULL, -1, methods
};
PyMODINIT_FUNC PyInit__hcu_hook_alias_scan(void) {
    return PyModule_Create(&definition);
}
"""


class NativeCacheError(RuntimeError):
    pass


def load():
    """Return the native scanner or None when this build is unsupported.

    Existing cache corruption is an error; it must not look like a clean fallback.
    """
    if (
        sys.implementation.name != "cpython"
        or not sys.platform.startswith("linux")
        or sysconfig.get_config_var("Py_GIL_DISABLED")
    ):
        return None
    import fcntl
    compiler = shlex.split(sysconfig.get_config_var("CC") or "cc")
    resolved = shutil.which(compiler[0])
    include = sysconfig.get_path("include")
    if not resolved or not (Path(include) / "Python.h").is_file():
        return None
    compiler[0] = resolved
    version = subprocess.check_output(compiler + ["--version"], text=True, timeout=10).splitlines()[0]
    identity = {"source": C_SOURCE, "abi": sysconfig.get_config_var("SOABI"),
                "cache_tag": sys.implementation.cache_tag,
                "platform": sysconfig.get_platform(), "compiler": compiler, "version": version}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    base = Path.home() / ".cache/hcu_adapter/hook_aliases"
    base.mkdir(parents=True, exist_ok=True)
    target = base / key
    with (base / (key + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not target.exists():
            stage = Path(tempfile.mkdtemp(prefix="stage-", dir=base))
            try:
                source = stage / "scan.c"
                source.write_text(C_SOURCE)
                library = stage / "scan.so"
                subprocess.run(compiler + ["-shared", "-fPIC", "-O3", "-std=c11",
                    "-I" + include, str(source), "-o", str(library)],
                    check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
                manifest = {"key": key, "sha256": hashlib.sha256(library.read_bytes()).hexdigest()}
                (stage / "manifest.json").write_text(json.dumps(manifest))
                os.rename(stage, target)
            finally:
                if stage.exists():
                    shutil.rmtree(stage)
        try:
            manifest = json.loads((target / "manifest.json").read_text())
            actual = hashlib.sha256((target / "scan.so").read_bytes()).hexdigest()
            if manifest != {"key": key, "sha256": actual}:
                raise NativeCacheError("hook alias cache hash mismatch")
        except (OSError, ValueError) as exc:
            raise NativeCacheError("hook alias cache is incomplete") from exc
    spec = importlib.util.spec_from_file_location("_hcu_hook_alias_scan", target / "scan.so")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.scan
