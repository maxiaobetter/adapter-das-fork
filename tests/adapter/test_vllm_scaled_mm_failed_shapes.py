# SPDX-License-Identifier: Apache-2.0
"""Metadata admission for shapes whose native provider leaves output unwritten."""
import math
import pytest
import torch
from hcu_adapter.kernels.scaled_mm_lightop import supported

class Metadata:
    def __init__(self,shape,dtype):
        self.shape=shape;self.dtype=dtype;self.ndim=len(shape)
        self.device='cuda:0';self.is_cuda=True
    def numel(self):return math.prod(self.shape)
    def is_contiguous(self):return True
    def t(self):return Metadata(self.shape[::-1],self.dtype)

@pytest.mark.parametrize('m',[112,128])
@pytest.mark.parametrize('k',[3072,6144,8192,12288])
@pytest.mark.parametrize('dtype',[torch.int8,torch.float8_e4m3fn])
def test_native_failed_int8_shapes_retain_official_math(m,k,dtype):
    a=Metadata((m,k),dtype);b=Metadata((k,6144),dtype)
    sa=Metadata((m,1),torch.float32);sb=Metadata((6144,1),torch.float32)
    assert supported(a,b,sa,sb,torch.bfloat16,None,'gfx938') is (dtype!=torch.int8)

def test_qualified_int8_hidden_projection_keeps_native():
    a=Metadata((128,6144),torch.int8);b=Metadata((6144,9216),torch.int8)
    sa=Metadata((128,1),torch.float32);sb=Metadata((9216,1),torch.float32)
    assert supported(a,b,sa,sb,torch.bfloat16,None,'gfx938')
