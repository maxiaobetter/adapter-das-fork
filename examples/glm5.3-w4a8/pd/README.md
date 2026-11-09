# GLM5.3 CT W4A8 独立 P/D 启动脚本

`prefill.sh`、`decode.sh` 各自包含必要的 export 和一条启动命令，直接编辑、运行。
不读取 `common.sh`、runtime shell 或 adapter YAML，不新增配置加载机制。

```bash
# 在配方目录运行；每组两台节点分别填写 rank 0 / 1。
bash prefill.sh --dist-init-addr PREFILL_LEADER:30941 --node-rank 0
bash decode.sh --dist-init-addr DECODE_LEADER:30942 --node-rank 0
```

将 `PREFILL_LEADER` / `DECODE_LEADER` 换成实际地址；也可直接修改脚本中的
`<prefill_node0_ip>` / `<decode_node0_ip>` 和 `--node-rank`。P/D 互不读取配置，
P 需要本目录的 `ep_config.json`。两端请求路由由外部 router/gateway 配置。

模型填写 [hygon/GLM-5.3-Channel-INT4-w4a8](https://www.modelscope.cn/models/hygon/GLM-5.3-Channel-INT4-w4a8)，
`export SGLANG_USE_MODELSCOPE=1` 选择下载源；本地权重可直接替换 `--model-path`。
公开配置已核对为 CT channel INT4 + dynamic token INT8，尚未核对与旧 checkpoint 的全部权重字节。

`python3 -u -m hcu_sglang` 是 plugin 的轻量启动入口，补齐 AITER 导入兼容后
转交官方 `sglang.launch_server`。Python 先经已安装 adapter 的 `sitecustomize.py`
建立适配层，官方再根据已安装包的 entry point 加载 `hcu_sglang:register`。
镜像须安装官方 `release/v0.5.22`、adapter、plugin 与配套算子库；只设
`PYTHONPATH` 不能保证 plugin 的安装元数据可发现。

| 角色 | 保留参数 |
| --- | --- |
| Prefill | 两节点 PP2 / TP8 / EP8 / CP8，DeepEP normal，LayerSplit，显存比例 0.8，关闭 graph |
| Decode | 两节点 TP16 / DP16 / EP16，DeepEP low_latency，显存比例 0.858，EAGLE 5/1/6，decode graph batch 16 |

保留 W4A8、INT8 激活、CUDA 平台语义与上下文长度 1048576。评测每题最大生成
100000 token 是评测端参数。P 的 PP2 不启用 MTP。

脚本顶部仅显式保留当前配方需要的选择与角色设置；当前 plugin 的专项默认尚未
统一迁移，不能简单删除这些 export。已有默认如 `HCU_W4A8_MOE=native`、
`SGLANG_USE_AITER=0`、`SGLANG_ROCM_USE_MULTI_STREAM=0`、K 反量化块长 16
不重复设置；`SGLANG_DSA_FUSE_TOPK=1` 用于覆盖 adapter 的旧模型自动规则。
`SGLANG_HCU_COMPACT_INDEX_BROADCAST` 已默认开启，P 脚本省去重复的 `=1`；
当前 GLM 专家路由已由 `SGLANG_USE_LIGHTOP=1` 选择 LightOp，P/D 均省去重复的
`SGLANG_LIGHTOP_TOPK=1`。`SGLANG_HCU_LAYER_SPLIT_SYNC_PREFETCH=0` 仍显式保留，
因为当前 plugin 默认是 1；本次不改变预取 stream 的选择。
旧 `SGLANG_ENABLE_HCU_CONCAT_MLA_ABSORB_Q`、`SGLANG_DSA_MQA_SPLIT_BY_SEQ`
在当前代码未找到消费者，未搬入。脚本内的 export 是显式配方值，调整时直接编辑对应行。

两份脚本显式保留原配方的 GLIBC/HIP/HSA runtime 设置：GLIBC TLS 空间、
HIP kernel event/batch、blit copy、kernarg pool、AQL queue、graph dispatch 和
all-reduce stream；不再假定这些参数已经固化进尚未构建的镜像。两个服务超时设置
继续保留。`HIP_GRAPH_USE_CMD_CACHE` 为 P=0/D=1，D 的硬件队列数为 3。

不写死 IB/SHCA/MLX 设备、网口、GID、网络 provider、动态库或源码安装路径。
`NCCL_IB_HCA`、`ROCSHMEM_ALLOWED_IBV_DEVICES`、`MC_TE_FILTERS` 及 socket 网口
由部署环境提供，不引用未定义的 `IB_DEVICES` / `SOCKET_IFNAME`。原环境其余网络
参数也须随实际 RCCL、DeepEP、Mooncake 版本和拓扑准备并核验，不能直接等同库默认。
Ninja、驱动/动态库和缓存目录仍由镜像准备；新组合未部署核验前，这些脚本不构成
原环境完整复现。已继承的网卡限制仍由部署方检查。

命令形式参考 [HYGON-AI cookbook](https://github.com/HYGON-AI/inference-cookbook-das/blob/7e8ce3c07d60926fbe86c29d923411a631b8dfaf/docs/model-deployment/sglang/glm-5.3.md)，
性能、精度以 sglang-das 0.5.18 为对照。公开 cookbook 的 FP8 配方只参考格式，
不替代本任务的 W4A8。`tools/hcu_config_ui.html`、`docs/config.md` 保持既有实现；
本次没有扩展配置系统。

本次仅做脚本与文档整理、静态检查。官方框架保持
`baa9befa381511cd4d56f83980951e5a45fffc6d`，没有提交、推送或创建 PR。
节点当前不可达；新脚本、新组合的启动、自动网卡发现、精度和性能均未复测。
