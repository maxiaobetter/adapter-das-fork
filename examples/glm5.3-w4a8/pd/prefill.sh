# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# 原运行配方的 HCU runtime 设置；镜像尚未固化，启动前显式保留。
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export HIP_GRAPH_ACCUMULATE_DISPATCH=1
export ALLREDUCE_STREAM_WITH_COMPUTE=1

# adapter / plugin 接入及当前 W4A8 配方需要的非默认选项。
export HCU_ADAPTER_HOST=sglang
export SGLANG_PLUGINS=hcu
export HCU_SLIMQUANT=0
export HCU_DEEPEP_BRIDGE=0
export HCU_RMS_NORM_IMPL=lightop
export HCU_FUSED_ADD_RMS_IMPL=lightop
export SGLANG_HCU_USE_LIGHTOP_ROPE=1
export SGLANG_HCU_W8A8_USE_LIGHTOP=1
export SGLANG_HCU_GLM_CT_BF16_ROUTER=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_ROCM_USE_AITER_MOE=0
export SGLANG_DSA_FUSE_TOPK=1
export SGLANG_OPT_USE_TOPK_V2=0
export SGLANG_DSA_HCU_INT8_INDEX_K_CACHE=1
export SGLANG_USE_DEEPGEMM_MOE=1
export SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_UVICORN_WORKER_HEALTHCHECK_TIMEOUT=120
export OMP_NUM_THREADS=1

# Prefill：CP8、normal DeepEP、LayerSplit。
export SGLANG_ENABLE_CP_V2=1
export SGLANG_DSA_HCU_REUSE_SORTED_TOPK=1
export SGLANG_USE_LIGHTOP_W4A8_MARLIN_MOE=0
export SGLANG_USE_W4A8_CONTIGUOUS_HIPC=1
export SGLANG_HCU_LAYER_SPLIT_SYNC_PREFETCH=0
export HIP_GRAPH_USE_CMD_CACHE=0

export SGLANG_USE_MODELSCOPE=1

python3 -u -m hcu_sglang \
    --model-path hygon/GLM-5.3-Channel-INT4-w4a8 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 30841 \
    --dist-init-addr "<prefill_node0_ip>:30941" \
    --nnodes 2 \
    --node-rank 0 \
    --pp-size 2 \
    --disable-overlap-schedule \
    --tp-size 8 \
    --dp-size 1 \
    --ep-size 8 \
    --moe-dense-tp-size 1 \
    --moe-a2a-backend deepep \
    --deepep-mode normal \
    --attn-cp-size 8 \
    --enable-prefill-cp \
    --cp-strategy interleave \
    --reasoning-parser glm45 \
    --tool-call-parser glm47 \
    --context-length 1048576 \
    --kv-cache-dtype fp8_e4m3 \
    --dtype bfloat16 \
    --mem-fraction-static 0.8 \
    --chunked-prefill-size 32768 \
    --max-prefill-tokens 32768 \
    --max-running-requests 96 \
    --disable-cuda-graph \
    --page-size 64 \
    --dsa-prefill-backend flashmla_sparse \
    --dsa-decode-backend flashmla_kv \
    --quantization compressed-tensors \
    --moe-runner-backend deep_gemm \
    --enable-cache-report \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend mooncake \
    --disaggregation-bootstrap-port 30949 \
    --json-model-override-args '{"index_share_for_mtp_iteration": false}' \
    --enable-dsa-cache-layer-split \
    --deepep-config ./ep_config.json \
    "$@"
