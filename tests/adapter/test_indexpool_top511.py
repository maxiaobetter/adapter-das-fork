# SPDX-License-Identifier: Apache-2.0
"""Portable Top511 contracts: actual Triton postprocessing runs in interpreter.

Only the public vendor native selection is doubled; no GPU result is claimed.
"""
from types import ModuleType, SimpleNamespace
import sys

import pytest
import torch


class CpuHcuTensor(torch.Tensor):
    @property
    def is_cuda(self):
        return True


def hcu(value):
    return value.as_subclass(CpuHcuTensor)


@pytest.fixture
def api(monkeypatch):
    import numpy as np
    import triton.language as tl
    from triton.runtime.interpreter import InterpretedFunction, InterpreterBuilder, TensorHandle
    from triton.runtime.jit import JITFunction
    from hcu_adapter.kernels import indexer_topk_lightop as module
    def function(*args, **kwargs):
        implementation = getattr(module, 'select_indexpool_groups', None)
        assert callable(implementation), 'Missing public IndexPool Top511 provider'
        return implementation(*args, **kwargs)
    # Match the existing portable interpreter repair in the indexer CPU suite.
    ops = (np.less, np.less_equal, np.greater, np.greater_equal, np.equal, np.not_equal)
    sample = InterpreterBuilder().create_icmpSLT(TensorHandle(np.array([0],dtype=np.int32),tl.int32), TensorHandle(np.array([1],dtype=np.int32),tl.int32))
    if sample.data.dtype != np.bool_ or sample.dtype != tl.int1:
        original = InterpreterBuilder.binary_op
        def binary_op(self, left, right, op):
            return TensorHandle(op(left.data,right.data),tl.int1) if op in ops else original(self,left,right,op)
        monkeypatch.setattr(InterpreterBuilder,'binary_op',binary_op)
    monkeypatch.setattr(JITFunction,'__call__',lambda self,*a,**kw:InterpretedFunction(self.fn)(*a,**kw))
    monkeypatch.setattr(module,'_topk_live_scores',InterpretedFunction(module._topk_live_scores.fn))
    monkeypatch.setattr(module,'_topk_native_output',InterpretedFunction(module._topk_native_output.fn))
    post = getattr(module, '_indexpool_top511_output', None)
    if post is not None:
        monkeypatch.setattr(module,'_indexpool_top511_output',InterpretedFunction(post.fn))
    monkeypatch.setattr(module,'_supported_device',lambda index:True)
    monkeypatch.setattr(torch.version,'hip','portable-cpu-contract')
    monkeypatch.setattr(torch.cuda,'get_device_properties',lambda device:SimpleNamespace(multi_processor_count=64))
    monkeypatch.delenv('HCU_DISABLE_OPS',raising=False)
    calls = []
    def native(scores, lengths, *, out_offsets, out_indices, row_starts=None):
        assert scores.dtype == torch.float32 and scores.stride(1) == 1
        assert lengths.dtype == out_offsets.dtype == out_indices.dtype == torch.int32
        assert out_indices.shape == (scores.shape[0],512) and row_starts is None
        assert bool((out_offsets==0).all())
        calls.append({'lengths':lengths.clone(),'pitch':scores.shape[1]})
        out_indices.fill_(-1)
        for row,length in enumerate(lengths.tolist()):
            # Independent native ABI model: order encoded FP32 keys, lower ID ties.
            raw=scores[row,:length].view(torch.int32).to(torch.int64)&0xffffffff
            key=torch.where((raw&0x80000000)!=0,(~raw)&0xffffffff,raw^0x80000000)
            chosen=torch.argsort(key,descending=True,stable=True)[:512].int()
            out_indices[row,:chosen.numel()].copy_(chosen.flip(0)) # unordered public output
    root=ModuleType('lightop');root.__path__=[]
    vendor=ModuleType('lightop.fuse_topk_transform');vendor.topk_transform_ragged_v2=native
    monkeypatch.setitem(sys.modules,'lightop',root);monkeypatch.setitem(sys.modules,'lightop.fuse_topk_transform',vendor)
    yield SimpleNamespace(module=module,select=function,calls=calls,vendor=vendor)
    assert not torch.cuda.is_initialized()


def oracle(scores, upper):
    rows=[]
    for values,length in zip(scores.tolist(),upper.tolist()):
        # Preserve original rank semantics; -Inf is absent, NaNs precede +Inf.
        eligible=[i for i in range(max(0,min(length,len(values)))) if values[i]!=-float('inf')]
        eligible.sort(key=lambda i:(0 if values[i]!=values[i] else 1,0 if values[i]!=values[i] else -values[i],i))
        selected=sorted(eligible[:511]);rows.append(selected+[-1]*(511-len(selected)))
    return torch.tensor(rows,dtype=torch.int32)


@pytest.mark.parametrize('width', [1,511,512,513,1029])
def test_exact_rank_and_causal_length_boundaries(api,width):
    # Logical IDs are adversarial to a naive unordered/native512 slice.
    backing=torch.empty((4,width+7),dtype=torch.float32)
    scores=backing[:,:width];scores.copy_(torch.arange(width).float().flip(0)[None,:])
    upper=torch.tensor([0,min(width,511),min(width,512),width],dtype=torch.int64)
    result=api.select(hcu(scores),upper,511)
    assert torch.equal(result,oracle(scores,upper))
    assert api.calls[-1]['lengths'].tolist()==upper.tolist()
    assert api.calls[-1]['pitch']>=512


def test_wide_pitched_rows_do_not_select_poisoned_tail(api):
    scores=torch.full((2,262148),float('nan'))[:,:262144]
    scores[:,:513]=torch.arange(513).float()[None,:]
    upper=torch.tensor([-1,513],dtype=torch.int32)
    result=api.select(hcu(scores),upper,511)
    assert torch.equal(result,oracle(scores,upper))
    assert api.calls[-1]['lengths'].tolist()==[0,513]


def test_nan_infinity_signed_zero_and_boundary_ties(api):
    scores=torch.cat((torch.tensor([float('nan'),float('inf'),-float('inf'),-0.,0.,torch.finfo(torch.float32).tiny,-torch.finfo(torch.float32).tiny]),torch.zeros(1020))).repeat(2,1)
    upper=torch.tensor([512,scores.shape[1]],dtype=torch.int32)
    result=api.select(hcu(scores),upper,511)
    assert torch.equal(result,oracle(scores,upper))
    assert not bool((result==2).any()), '-Inf must remain absent'


def test_fp32_ranks_are_not_rounded_to_bf16(api):
    # All512 values would round to one BF16 bucket; the lowest score has ID0.
    scores=1.+torch.arange(512,dtype=torch.float32)[None,:]*2**-20
    upper=torch.tensor([512],dtype=torch.int32)
    result=api.select(hcu(scores),upper,511)
    assert torch.equal(result,torch.arange(1,512,dtype=torch.int32)[None,:])


@pytest.mark.parametrize('budget',[0,7,512,1024])
def test_unsupported_budgets_do_not_invoke_native(api,budget):
    assert api.select(hcu(torch.ones(1,1024)),torch.tensor([1024],dtype=torch.int32),budget) is None
    assert not api.calls


@pytest.mark.parametrize('kind',['cpu','dtype','column-stride','length-dtype','length-shape','empty-rows','empty-width'])
def test_unsupported_metadata_retains_fallback(api,kind):
    scores=hcu(torch.ones(1,1024));upper=torch.tensor([1024],dtype=torch.int32)
    if kind=='cpu':scores=torch.ones(1,1024)
    elif kind=='dtype':scores=scores.bfloat16()
    elif kind=='column-stride':scores=scores[:,::2]
    elif kind=='length-dtype':upper=upper.float()
    elif kind=='empty-rows':scores=scores[:0];upper=upper[:0]
    elif kind=='empty-width':scores=scores[:,:0]
    else:upper=upper[:,None]
    assert api.select(scores,upper,511) is None
    assert not api.calls


def test_callable_missing_native_sentinel_falls_back(api):
    from hcu_adapter._base import Missing
    api.vendor.topk_transform_ragged_v2=Missing('lightop.fuse_topk_transform.topk_transform_ragged_v2','lightop')
    assert callable(api.vendor.topk_transform_ragged_v2)
    assert api.select(hcu(torch.ones(1,512)),torch.tensor([512],dtype=torch.int32),511) is None
    assert not api.calls


def test_absent_native_export_falls_back(api):
    del api.vendor.topk_transform_ragged_v2
    assert api.select(hcu(torch.ones(1,512)),torch.tensor([512],dtype=torch.int32),511) is None


def test_real_native_exception_propagates(api):
    def fail(*a,**kw):raise RuntimeError('native launch failed')
    api.vendor.topk_transform_ragged_v2=fail
    with pytest.raises(RuntimeError,match='native launch failed'):
        api.select(hcu(torch.ones(1,512)),torch.tensor([512],dtype=torch.int32),511)


def test_fp32_nan_payload_sign_and_subnormal_keys(api):
    bits=torch.tensor([-2147483648,0,-8388608,2139095040,2143289344,-4194304,1,-2147483647],dtype=torch.int32)
    scores=torch.cat((bits.view(torch.float32),torch.linspace(-1,1,1024)))[None,:]
    upper=torch.tensor([scores.shape[1]],dtype=torch.int32)
    assert torch.equal(api.select(hcu(scores),upper,511),oracle(scores,upper))


def test_registered_quick_off_retains_torch_fallback(api,monkeypatch):
    from hcu_adapter.infra import kill_switch
    assert 'hy4_indexpool_topk' in kill_switch.SWITCHES
    monkeypatch.setenv('HCU_DISABLE_OPS','hy4_indexpool_topk')
    assert api.select(hcu(torch.ones(1,512)),torch.tensor([512],dtype=torch.int32),511) is None
    assert not api.calls


def test_existing_dsv41_ragged512_ordering_is_unchanged(api):
    scores=torch.arange(1029).float()[None,:].expand(2,-1).clone()
    starts=torch.zeros(2,dtype=torch.int32);ends=torch.tensor([500,800],dtype=torch.int32)
    result=api.module.select_topk(hcu(scores),starts,ends,512,valid_lengths=ends)
    assert torch.equal(result[0],torch.cat((torch.arange(500,dtype=torch.int32),torch.full((12,),-1,dtype=torch.int32))))
    assert torch.equal(result[1],torch.arange(799,287,-1,dtype=torch.int32))
