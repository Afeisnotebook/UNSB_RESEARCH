# 本地论文头图动机搜索：最终状态

本目录记录一次纯基线、无方法分支的本地动机搜索。唯一比较对象是：

```text
五个 task-specific plain UNSB
            vs
一个五域共享 All-in-One plain UNSB
```

最终支持的观测不是“ AIO 更不确定”，而是 **shared-bridge domain-phase desynchronization**：同一个 AIO epoch-1 checkpoint，在固定 bridge time 上最接近不同域不同年龄的 task-specific conditional transition kernel。

域顺序固定为 Fog / Low-light / Rain / Rain-streak / Snow，三处 bridge time 的有效年龄为：

```text
t=0.50: e4 / e3 / e2 / e4 / e5
t=0.74: e4 / e3 / e2 / e4 / e5
t=0.86: e4 / e3 / e2 / e2 / e5
```

该 map 在三个互不重叠的图像 split 上完全一致：20、24、16 图/域。第三 split 在读取 effect 前冻结 map 和门槛，最终得到：

- mapping accuracy：15/15；
- within-time mapping permutation：`p=0.0002`；
- bootstrap-stable predicted cells：14/15；
- M16/M32：15/15；
- verdict：`SUPPORTED_DOMAIN_PHASE_MAPPING_REPLICATION`。

## 必须同时阅读的反证

1. 初始方向离散在历史六 seed 上稳定复现，但 fresh dual-control 只得到 `EXPOSURE_ONLY`；optimizer-clock 和 Single e1–e5 age envelope 不支持更强结论。
2. AIO 条件平均方向没有稳定越出 Single e1–e5 轨迹，裁决为 `NOT_DISTINCT_FROM_SINGLE_AGE_TRAJECTORY`。
3. 第一次 phase confirmation 的 raw map 与发现集 15/15 一致，但原 null 错误地跨不同域专用参考模型移动整条 KDD profile，裁决永久更正为 `INVALID_NULL_EXCHANGEABILITY`。修复后的第三 split 只打乱预定年龄对固定域身份的指派。
4. discovery selector 的 stem namespace 曾错误，100 张 discovery 图中有 14 张与历史测量身份重合；它只作为发现 evidence。后两个 split 使用 canonical stem，历史 overlap 为 0。

## 论文表述边界

允许：在本地五域制度、seed=2041 下，一个 AIO e1 checkpoint 对应跨域不同的 task-specific conditional-kernel phases；固定 map 在三个零重叠图像 split 上复现。

禁止：多训练 seed 确认、external/sealed confirmation、RainDS-syn 覆盖、校准 posterior uncertainty、因果恢复伤害，或已经证明 phase correction 会提升最终质量。

## 文件路由

- 冻结协议：`HEADFIGURE_FROZEN_PROTOCOL.json`、`RECIPROCAL_KERNEL_FROZEN_PROTOCOL.json`、`PHASE_CLOCK_CONFIRMATION_FROZEN_PROTOCOL.json`、`PHASE_MAPPING_RECONFIRM_FROZEN_PROTOCOL.json`。
- 更正记录：`PHASE_NULL_INVALIDATION_RECORD_CN.md`、`DISCOVERY_STEM_NORMALIZATION_CORRECTION.json`。
- 执行代码：`../code/headfigure_local/`。
- 仓库外 discovery run：`E:/UNSB_Expl/UNSB_HEADFIGURE_LOCAL_20260824/`。
- 仓库外 first confirmation：`E:/UNSB_Expl/UNSB_HEADFIGURE_PHASE_CONFIRM_20260824/`。
- 仓库外 tertiary confirmation 与最终图文：`E:/UNSB_Expl/UNSB_HEADFIGURE_PHASE_TERTIARY_20260824/`。
- 可迁移交付副本：`final_delivery/`。
