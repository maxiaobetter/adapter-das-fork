"""Native IPC UUID lookup must follow the worker's physical GPU mapping."""
import os
import sys
import types
import pytest
from hcu_adapter.bridges import vllm as bridge

@pytest.mark.parametrize('visible,logical,expected',[
    ('',0,0),('',7,7),('3',0,3),('7,2,5',1,2),('0,1,2,3,4,5,6,7',6,6)])
def test_uuid_uses_physical_device(monkeypatch,visible,logical,expected):
    class Rocm:
        @classmethod
        def device_id_to_physical_device_id(cls,index):
            text=os.environ.get('HIP_VISIBLE_DEVICES','')
            return int(text.split(',')[index]) if text else index
        @classmethod
        def get_device_uuid(cls,physical=0):return 'physical-uuid-%d'%physical
    rocm=types.ModuleType('vllm.platforms.rocm');rocm.RocmPlatform=Rocm
    monkeypatch.setitem(sys.modules,'vllm.platforms.rocm',rocm)
    monkeypatch.delitem(bridge.__dict__,'HcuPlatform',raising=False)
    monkeypatch.setenv('HIP_VISIBLE_DEVICES',visible)
    monkeypatch.delenv('ROCR_VISIBLE_DEVICES', raising=False)
    platform=bridge._hcu_platform_class()
    assert platform.get_device_uuid(logical)=='physical-uuid-%d'%expected
    bridge.__dict__.pop('HcuPlatform',None)


@pytest.mark.parametrize("mask,expected", [
    ("GPU-0000000000000017", [1]),
    ("GPU-0000000000000017,0", [1, 0]),
    ("0,GPU-0000000000000017", [0, 1]),
    ("1,0", [1, 0]),
    ("GPU-000000000000001", [1]),
    ("GPU-000000000000001,0", [1, 0]),
    ("1,GPU-00000000000012", [1, 0]),
])
@pytest.mark.parametrize("with_enumeration", [True, False])
def test_rocr_mask_resolves_smi_indices(monkeypatch, mask, expected, with_enumeration):
    rocm = types.ModuleType("vllm.platforms.rocm")
    rocm.with_amdsmi_context = lambda function: function
    smi = types.ModuleType("amdsmi")
    smi.AmdSmiException = type("AmdSmiException", (Exception,), {})
    smi.amdsmi_get_processor_handles = lambda: ("card0", "card1")
    serials = {"card0": "0x1234", "card1": "0x0017"}
    smi.amdsmi_get_gpu_asic_info = lambda handle: {"asic_serial": serials[handle]}
    if with_enumeration:
        smi.amdsmi_get_gpu_enumeration_info = lambda handle: {
            "hip_uuid": f"GPU-{int(serials[handle], 16):016X}"
        }
    monkeypatch.setitem(sys.modules, "amdsmi", smi)
    monkeypatch.setitem(sys.modules, "vllm.platforms.rocm", rocm)
    assert bridge._rocr_physical_ids(mask) == expected
    with pytest.raises(ValueError, match="duplicate GPU indices"):
        bridge._rocr_physical_ids("GPU-0000000000000017,1")
    with pytest.raises(ValueError, match="missing or ambiguous"):
        bridge._rocr_physical_ids("GPU-000000000000abcd")

    with pytest.raises(ValueError, match="duplicate GPU indices"):
        bridge._rocr_physical_ids("GPU-000000000000001,1")
    for invalid in ("GPU-0", "GPU-", "GPU-00000000000000170", "000000000000001a"):
        with pytest.raises(ValueError, match="missing or ambiguous"):
            bridge._rocr_physical_ids(invalid)
