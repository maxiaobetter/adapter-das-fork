#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
: "${MODEL_PATH:?Set MODEL_PATH to the GLM-5.3-Flash blockwise FP8 checkpoint}"
: "${SGLANG_CHECKOUT:?Set SGLANG_CHECKOUT to the SGLang v0.5.22 source checkout}"
: "${DIST_INIT_ADDR:?Set DIST_INIT_ADDR to the node 0 address and rendezvous port}"
: "${NODE_RANK:?Set NODE_RANK to 0 or 1}"
case "$NODE_RANK" in
    0|1) ;;
    *) printf 'NODE_RANK must be 0 or 1\n' >&2; exit 2 ;;
esac
case "${DETERMINISTIC:=1}" in
    0|1) ;;
    *) printf 'DETERMINISTIC must be 0 or 1\n' >&2; exit 2 ;;
esac
if [[ -n "${ADAPTER_CHECKOUT:-}" ]]; then
    adapter_dir="$(cd -- "$ADAPTER_CHECKOUT" && pwd -P)"
else
    adapter_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
fi
if [[ ! -d "$adapter_dir/hcu_adapter" ]]; then
    printf 'Set ADAPTER_CHECKOUT to the adapter repository root
' >&2
    exit 2
fi
if [[ ! -r "$SGLANG_CHECKOUT/python/sglang/launch_server.py" || ! -r "$MODEL_PATH/config.json" ]]; then
    printf 'Check SGLANG_CHECKOUT and MODEL_PATH\n' >&2
    exit 2
fi
export PYTHONPATH="$SGLANG_CHECKOUT/python:$adapter_dir${PYTHONPATH:+:$PYTHONPATH}"
export HCU_ADAPTER_HOST=sglang SGLANG_PLUGINS=none
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-$NCCL_SOCKET_IFNAME}"
export HCU_ORDERED_TP8_BF16_REDUCE="$DETERMINISTIC"
export HCU_KDA_FIXED_SAFE_GATE_ANCHOR="$DETERMINISTIC"
export HCU_GLM53_FIXED_KDA_PROJECTION="$DETERMINISTIC"
export HCU_FIXED_BLOCK_FP8_DENSE="$DETERMINISTIC"
export HCU_DSA_ORDERED_SHORT_TOPK="$DETERMINISTIC"
export HCU_GLM53_FIXED_MHC_STATS="$DETERMINISTIC"
# The gfx938 fixture uses the unfused mHC pre/post path for both arms.
export SGLANG_OPT_USE_TILELANG_MHC_PRE="${SGLANG_OPT_USE_TILELANG_MHC_PRE:-false}"
export SGLANG_OPT_USE_TILELANG_MHC_POST="${SGLANG_OPT_USE_TILELANG_MHC_POST:-false}"
export HCU_GLM53_FIXED_EAGLE_PROJECTIONS="$DETERMINISTIC"
export HCU_GLM53_FIXED_DSA_MATMUL="$DETERMINISTIC"

exec python3 -u -m sglang.launch_server \
    --model-path "$MODEL_PATH" --served-model-name glm53-flash-blockfp8 \
    --host "${BIND_HOST:-127.0.0.1}" --port "${PORT:-25832}" \
    --tp-size 8 --nnodes 2 --node-rank "$NODE_RANK" --dist-init-addr "$DIST_INIT_ADDR" \
    --disable-custom-all-reduce --context-length 8192 --chunked-prefill-size 4096 \
    --max-running-requests 8 --max-total-tokens 32768 --max-mamba-cache-size 128 \
    --cuda-graph-backend-decode full --cuda-graph-max-bs-decode 2 --cuda-graph-bs-decode 1 2 \
    --disable-prefill-cuda-graph --speculative-algorithm EAGLE --speculative-num-steps 5 \
    --speculative-eagle-topk 1 --speculative-num-draft-tokens 6 --skip-server-warmup
