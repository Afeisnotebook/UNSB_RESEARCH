# 确定性修复审计（阶段1）

## 结论

阶段1通过：同 seed=2026、同参数、3 epoch 训练两次，`3_net_G.pth` SHA256 完全一致。

## 修复内容

1. `models/det_pad.py`：`DeterministicReflectionPad2d`，用 slice/index-select+cat 实现反射 padding。
   - forward 与 `F.pad(mode='reflect')` 逐位一致；
   - backward 不依赖 `reflection_pad2d_backward_cuda`；
   - `models/networks.py` / `models/ncsn_networks.py` 已全部替换。
2. `train.py` / `test.py`：
   - 在 import torch 前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`；
   - `--seed>=0` 时启用 `torch.use_deterministic_algorithms(True)`；
   - 保留并补全 `cudnn.deterministic=True`、`benchmark=False`、TF32 off。

## 验证证据

- CPU 单测：`pytest -q tests/test_det_pad.py` 4/4 通过。
- 同 seed 两次 3-epoch GPU smoke：`smoke_det_e` 与 `smoke_det_f`。

```text
sha256(3_net_G.pth):
smoke_det_e  4675f645675a038d7bd25ce9dee0d57e917c437147b2d828aa6cd804cce4f0ef
smoke_det_f  4675f645675a038d7bd25ce9dee0d57e917c437147b2d828aa6cd804cce4f0ef
```

- loss 行 `End of epoch` 内容一致，仅 wall-time 秒数不同。

## 口径

- 旧 `reflection_pad2d` 非确定性问题已消除；阶段2干净核心重跑采用该修复。
- 论文仍报告 mean±CI，并保留 CuBLAS/deterministic harness 的说明；不使用“不可约运行间方差”作为旧 limitation，除非另有证据。
