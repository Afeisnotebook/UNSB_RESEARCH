# UNSB 动机图纯基线重启（旁路任务）

本目录是一条与主线 `unsb_tired` 隔离的旁路任务，目标只有一个：

> 用**干净重构代码**（`unsb_tired/refactor` 的 plain `SBModel` + clean `DTCovMatch`）重建论文动机证据链，
> 研究“多域共享训练是否改变 UNSB 的条件恢复方向几何”。

## 唯一对照

```text
Single-task（5 个独立 plain UNSB）   vs   Plain All-in-One（1 个共享 plain UNSB）
```

`DT` 只作为后置的路径尺度干预 sanity check，`HJ` 不进动机图。

## 五个天气域（已去掉 RainDS-syn）

- FoggyCityscapes
- LowLightTrafficData
- RainCityscapes
- RSCityscapes
- SnowTrafficData

## 科学定调（继承 master 文档结论）

- 保留：路径尺度方向几何 b/c（唯一被多轮实验支持的现象）。
- 优化：去掉 RainDS-syn；用干净确定性实现；d/e 改为路径量 U 的空间分解 + 图像级差值分布。
- 取消：局部结构冲突 d/e 作为必要性主张；HJ 负尾修复主张；AIO 全程更分散的全称结论。

## 目录

```text
motivation_baseline_restart/
├─ README_CN.md
├─ PATH_MAP.json            # 路径映射（机器可读）
├─ CODE_IDENTITY.json       # 源码身份 SHA-256
├─ DATA_MANIFEST.json/.csv  # 五域 split 身份清单
├─ MOTIVATION_FROZEN_SPEC.json
├─ code/                    # prepare_data / measure / train queue
├─ datasets/                # 生成的五域 dataroot（硬链接）
├─ checkpoints/             # 训练 checkpoint
├─ raw/ figures/ reports/   # 证据 / 图 / 裁决报告
└─ MANIFEST.sha256
```

## 执行顺序

1. `code/discover.py`：只读生成 PATH_MAP / CODE_IDENTITY / DATA_MANIFEST。
2. `code/prepare_data.py`：按 DATA_MANIFEST 硬链接生成五域 dataroot。
3. `code/train_queue.sh`：在宿主机 GPU 上串行低优先级训练。
4. `code/run_measure.py`：从 checkpoint 离线测量 b/c/d/e。

> 注：本工作区所在的 Codex 沙箱没有 GPU 设备；训练与离线测量必须在宿主机 GPU 上执行。
