# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["count"])
def _sum_bf16_tp8(parts_ptr, output_ptr, count, BLOCK: tl.constexpr):
    count = count.to(tl.int64)
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < count
    accumulated = tl.load(parts_ptr + offsets, mask=valid, other=0)
    for rank in tl.static_range(1, 8):
        partial = tl.load(parts_ptr + rank * count + offsets, mask=valid, other=0)
        # Match an in-place BF16 add at each rank, including its rounding.
        accumulated = (accumulated.to(tl.float32) + partial.to(tl.float32)).to(
            tl.bfloat16, fp_downcast_rounding="rtne"
        )
    tl.store(output_ptr + offsets, accumulated, mask=valid)


def sum_bf16_tp8(parts, output):
    count = output.numel()
    _sum_bf16_tp8[(triton.cdiv(count, 256),)](parts, output, count, BLOCK=256, num_warps=4)
