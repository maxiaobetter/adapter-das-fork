# SGLang SlimQuant provider 归属

SGLang 选择已安装的 `hcu_sglang:register` 插件入口时，adapter 跳过 legacy
SlimQuant 的完整激活步骤，包括权重加载 shim 和量化注册，由 plugin 提供格式支持。
无需额外设置 `HCU_SLIMQUANT=0`；已有配方中的显式 `0` 仍关闭 legacy 实现。

判断遵循 `SGLANG_PLUGINS` 白名单，以及 `SGLANG_PLATFORM` 对未选中平台发行包的
排除规则，只读取安装元数据，不导入 plugin 或提前执行其注册入口。
未安装或未选择该 plugin 时，保留 adapter 原有激活行为。变量默认值仍为 `1`。
plugin 入口加载失败不表示可安全回退到 legacy；两者的权重加载和 MoE 路径不同。

CPU 回归覆盖入口发现、白名单、平台排除及两条实际激活分支。
该接线不修改 SlimQuant 数学实现、底层 GEMM 或 DeepEP 双 ABI 桥；
CPU 回归不代表目标分支组合的模型精度、性能或 GPU 验收通过。

2026-11-27：基于 `7ea3a711` 的候选通过 provider 4 项与目标已有 DeepEP
dispatch 3 项 CPU 回归，L0 通过（31 项警告）；未改运行服务。
