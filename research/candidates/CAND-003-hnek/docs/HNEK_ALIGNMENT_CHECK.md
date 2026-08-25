# HNEK adapter ↔ refactor/baseline 只读对齐验证（2026-08-17，未训练、未动 GPU）

> **历史对齐记录：** 接口对齐结论仍可参考；“未训练”状态已过时。后续搜索与 e200 结果见 [当前裁决](../../../../decisions/CURRENT.md)。

## 结论先行

`hnek_kernel` / `hnek_adapter` 是自包含的，**可以直接作用在 `refactor/baseline` 的
`SBModel` 上**，接口全部对齐，无需改公式、阈值、schedule 或科学语义。唯一断的是
**runner 的 harness 依赖**：`UNSB_Long/code/scripts/hnek/*.py` import 的 `scripts.final1`
不在同树，而在 `UNSB_EvidenceFirst_Rebuild_Bootstrap_20260806/scripts/final1/`。

## 1. 自包含 import（CPU 已验证）

- `hnek_kernel.py` 只依赖 `numpy` / `torch`，import OK。
- `hnek_adapter.py` 只依赖 `types` / `dataclasses` / `torch` 与 `hnek_kernel`，import OK。
- 无需 `scripts.final1`。

## 2. bridge_schedule 一致性（数值已验证）

```text
hnek bridge_schedule(5) : [0, 0.5, 0.74, 0.86, 0.94, 1.0]
refactor/baseline times  : [0, 0.5, 0.74, 0.86, 0.94, 1.0]
MATCH
```

两边的 schedule 构造公式完全一致。

## 3. SBModel 接口对齐（静态核对，全命中）

HNEK adapter 会重绑/引用以下对象，全部在 `refactor/baseline/models/sb_model.py` 存在：

```text
netG netE netD fake_B fake_B2 real_A_noisy real_A_noisy2 time_idx idt_B
loss_G_GAN loss_SB loss_NCE loss_NCE_Y loss_G criterionGAN calculate_NCE_loss
compute_E_loss compute_G_loss
```

- `netG.forward(x, time_cond, z, layers=[], encode_only=False)`：与 adapter 的
  `hnek_forward` 调用签名一致（`ResnetGenerator_ncsn.forward`）。
- `netE` 三参调用 `netE(input, time_idx, input2)`：与 `sb_model.py` 自身调用一致，
  adapter 只是把第二个 cat 的内容从 `y` 换成归一化残差 `r`（这是有意的坐标修正）。

## 4. full-state 恢复路径

- adapter 安装时带 `state-dict key` 与参数量不变性检查，**0 个新参数、不新增 state key**。
- 因此 `refactor/baseline` 自己的 warmup→train 检查点路径不受影响；适配层在模型构造/加载
  之后再安装即可。
- 注意：spec 里的 `coupled_common_randomness`（PLAIN/HNEK 共享同一 RNG bundle）和
  `full_state_e50.pt` 属于 `scripts.final1` 的 coupled harness，不是 adapter 的职责。
  若用 `refactor/baseline/train.py` 跑，需用“同 seed、同 warmup、分开训练”来近似 coupled，
  其严格程度略低于 final1 的共享 RNG bundle。

## 5. 断点定位

`UNSB_Long/code/scripts/hnek/run_hnek_decisive.py` 等 runner：

```python
from scripts.final1 import final1_common as C
from scripts.final1 import final1_networks as N
from scripts.final1 import final1_metrics as F1M
```

但 `UNSB_Long/code/scripts/` 下没有 `final1/`；真正的 final1 在
`UNSB_Long/UNSB_EvidenceFirst_Rebuild_Bootstrap_20260806/scripts/final1/`。
因此：
- 直接用 `UNSB_Long/code/scripts/hnek` 的 runner 会因为缺 `scripts.final1` 而失败；
- 但 `hnek_kernel` + `hnek_adapter` 本身不受影响。

## 6. 最小对齐（下一步，未执行）

要让 frozen 最小实验在 `refactor/baseline` 上直接启动，只需一个 thin shim：

1. 把 `hnek_kernel.py` / `hnek_adapter.py` 拷贝到 `refactor/baseline/models/hnek/`
   （或 `refactor/hnek/`），保持内容逐字节不变；
2. 在 `SBModel` 增加一个只读标志（如 `--hnek True`），构造后调用
   `install_hnek_model(model)`，不触碰公式/阈值/schedule；
3. 用 `refactor/baseline/train.py` 分别跑 PLAIN 与 HNEK，同 seed、同 warmup，
   development 集评测，再套用 `hnek_adjudicator` 的冻结门控。

这一步是“只读 import/API 适配”，不改科学语义；是否执行、是否采用近似 coupled（而非
final1 的共享 RNG bundle）需要你确认。
