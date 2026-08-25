from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DOMAINS = [
    "FoggyCityscapes",
    "LowLightTrafficData",
    "RainCityscapes",
    "RainDS-syn",
    "RSCityscapes",
    "SnowTrafficData",
]


def fmt_interval(values: list[float]) -> str:
    return f"[{values[0]:.4f}, {values[1]:.4f}]"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--existing-validation")
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    prepare = json.loads((root / "state" / "PREPARE_AUDIT.json").read_text(encoding="utf-8"))
    training = json.loads((root / "state" / "TRAINING_STATE.json").read_text(encoding="utf-8"))
    measurement = json.loads((root / "state" / "MEASUREMENT_STATE.json").read_text(encoding="utf-8"))
    stats = json.loads((root / "reports" / "PHASE_STATISTICS.json").read_text(encoding="utf-8"))
    cells = pd.read_csv(root / "reports" / "PHASE_CELL_SUMMARY.csv")
    previous = None
    if args.existing_validation:
        previous = json.loads(Path(args.existing_validation).read_text(encoding="utf-8"))

    maps = []
    result_rows = []
    for result in stats["time_results"]:
        ages = " / ".join(str(result["domain_effective_ages"][domain]) for domain in DOMAINS)
        maps.append(f"- `t={result['bridge_time_value']:.2f}`：`{ages}`")
        result_rows.append(
            "| {time:.2f} | {common} | {regret:.4f} | {ci} | {norm:.1f}% | {positive:.1f}% | {w2:.3f} | {reliability:.3f} |".format(
                time=result["bridge_time_value"],
                common=result["best_common_age"],
                regret=result["crossfit_sync_regret"],
                ci=fmt_interval(result["bootstrap_ci_95"]),
                norm=100 * result["crossfit_normalized_regret"],
                positive=100 * result["bootstrap_positive_fraction"],
                w2=result["wasserstein_barycenter_energy"],
                reliability=result["phase_reliability_ratio"],
            )
        )

    cell_rows = []
    for _, row in cells.iterrows():
        cell_rows.append(
            f"| {row['bridge_time_value']:.2f} | {row['domain']} | e{int(row['effective_age'])} | "
            f"e{int(row['effective_age_M16'])} | {row['bootstrap_modal_share'] * 100:.1f}% | "
            f"{row['profile_margin']:.4f} | {row['crossfit_domain_regret']:+.4f} |"
        )

    previous_section = ""
    if previous is not None:
        rows = []
        for result in previous["time_results"]:
            rows.append(
                f"| {result['bridge_time_value']:.2f} | {result['crossfit_sync_regret']:.4f} | "
                f"{fmt_interval(result['bootstrap_ci_95'])} | "
                f"{result['crossfit_normalized_regret'] * 100:.1f}% |"
            )
        previous_section = """
## 7. 与此前五域三 split 的关系

在冻结六域实验前，新统计量先在旧五域三个零重叠 split 的合并原始 KDD（60 图/域）上做了只读验算。它不是六域确认结果，也没有与新数据混成一个总体；其作用是排除“公式看起来漂亮但对已知现象没有分辨力”。

| t | 五域交叉拟合遗憾 | 95% 区间 | 占 profile range |
|---:|---:|---:|---:|
""" + "\n".join(rows) + "\n"

    report = f"""# UNSB 六域扩大版动机证据：共享桥相位场与共享时钟遗憾

> 研究范围：只比较 plain UNSB 的六个 task-specific Single 与一个六域 All-in-One；没有接入 DT、HJ 或其它候选算法，没有用 paired target 或恢复质量挑选现象。

## 1. 最终裁决

**`{stats['verdict']}`**。

固定域顺序为：`Fog / Low-light / Rain / RainDS / Rain-streak / Snow`。同一个六域 AIO epoch-1 checkpoint 在三个 bridge time 上投影到如下 task-specific 条件核相位：

{chr(10).join(maps)}

主裁决不是“这些整数看起来不同”，而是强迫所有域共用一个相位所产生的 held-out 条件核代价。冻结的四项判据为：

```json
{json.dumps(stats['requirements'], ensure_ascii=False, indent=2)}
```

M16/M32 有效相位一致为 **{stats['M16_M32_effective_age_agreement']}/{stats['M16_M32_total_cells']}**；domain-time phase shear energy 为 **{stats['domain_time_phase_shear_energy']:.4f}**。

## 2. 为什么先前是五域，现在恢复为六域

项目原始训练制度一直有六域，第六域是 `RainDS-syn`。2026-08-24 的上一版头图沿用“五域权威子协议”，目的是不把早期六域实验与后续五域统计直接拼接；它不是因为 RainDS-syn 缺数据或被科学否决。

本地原始身份数为：Fog 4575、Low-light 736、Rain 1188、RainDS 200、Rain-streak 1188、Snow 1266。六域均衡规模由只有 200 个身份的 RainDS 限制，因此本轮从头建立：

- 每域 120 trainA、120 unpaired trainB；
- 每域 80 held-out A，共 480 张观测图；
- 六个 Single 各训练 6 epoch；
- 一个 AIO 训练 1 epoch；
- Single e1 与 AIO e1 的本域期望曝光均为 120；Single e6 与 AIO e1 的总 optimizer step 均为 720。

五个旧域的 80 张新 held-out 均排除了此前三轮共 60 个 phase 身份；RainDS 首次进入这套相位测量。数据准备裁决为 `{prepare['verdict']}`。扩大样本只用于提高覆盖与精度，没有按结果重新选择图像。

## 3. 从离散现象到数学统计量

### 3.1 域专用条件核轨迹

记 AIO e1 在域 `d`、bridge time `t` 上与该域 Single age `e` 的互易条件方向距离为：

`K_d,t(e) = 1/2 [δ(mu_A(X_A), mu_S,e(X_A)) + δ(mu_A(X_S,e), mu_S,e(X_S,e))]`，

其中 `δ(u,v)=1-cos(u,v)`，`mu` 是 M=32 个随机 endpoint direction 单位化后的平均方向。它不是完整条件律之间的距离，而是一个直接落在 UNSB 条件转移方向上的可复算角距离。

域–时有效相位是 `phi_d,t = argmin_e K_d,t(e)`。这给出直观的 phase map，但单独看 argmin 会忽略谷底是否平、是否由样本噪声造成。

### 3.2 共享时钟遗憾

定义所有域被迫共用相位 `c` 时的额外错配：

`G_t(c) = (1/D) sum_d [K_d,t(c) - min_e K_d,t(e)]`，

以及最佳共享时钟仍无法消除的代价：

`G_t* = min_c G_t(c)`。

这正面回答：一个 global AIO checkpoint 若被解释成“所有域处在同一任务阶段”，相比允许域专用阶段，额外支付多少条件核几何代价。

为消除在同一批图上选谷底又评价谷底的偏差，每域 80 张图预先哈希分成 40/40。A 折拟合 `c` 和各域 `phi_d`，B 折评价；随后交换并平均。5000 次 bootstrap 每次分别重采样训练折和评价折，并重新拟合时钟。

### 3.3 它在局部等价于曲率加权相位方差

若 `K_d,t(e)` 在域专用最优相位附近二阶可微：

`K_d,t(c)-K_d,t(phi_d,t) ≈ 1/2 h_d,t (c-phi_d,t)^2`。

于是最佳共享相位近似为 `c_t* = sum_d h_d,t phi_d,t / sum_d h_d,t`，而：

`G_t* ≈ (1/(2D)) sum_d h_d,t (phi_d,t-c_t*)^2`。

所以共享时钟遗憾不是普通的“epoch 标签方差”，而是 **由条件核轨迹曲率加权的相位离散能量**：谷底很平的域不会仅因 argmin 差一格就被夸大，谷底尖锐且错位的域才产生真实代价。

### 3.4 相位 Wasserstein 离散

对每个域–时单元 bootstrap 图像并重新求 argmin，得到经验相位分布 `Pi_d,t`。定义：

`E_W(t) = (1/D) sum_d W2^2(Pi_d,t, Barycenter_W2(Pi_1,t,...,Pi_D,t))`。

一维 W2 barycenter 由各分布分位函数的平均直接计算，不需要人为 soft-min 温度。它衡量的是完整相位抽样分布的分离，而不仅是六个点估计。

## 4. 六域定量结果

| t | 最佳共享相位 | cross-fit regret | 95% bootstrap | 占 profile range | 正值 draws | W2 energy | reliability |
|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(result_rows)}

逐域单元：

| t | 域 | M32 phase | M16 phase | bootstrap mode share | KDD margin | cross-fit contribution |
|---:|---|---:|---:|---:|---:|---:|
{chr(10).join(cell_rows)}

## 5. 这个结果对论文意味着什么

1. **框架差异先于算法。** 证据来自同一个 plain UNSB 的 Single/AIO 对照，不是候选方法相对 baseline 的得分。
2. **对象属于 Schrödinger Bridge 过程。** 统计量建立在模型诱导 bridge states 上的随机 endpoint transition directions，而不是通用 loss、梯度冲突或 PSNR。
3. **指出了可优化的量。** 原来的 `phi_d,t` 只告诉我们“时钟不同”；`G_t*` 进一步告诉我们共享时钟为此支付的条件核代价。未来的 phase calibration 方法应预注册地降低 held-out `G_t*`，而不是只让热图更整齐。
4. **offset 与 shear 可以分开。** Wasserstein energy 表示同一 bridge time 的跨域 phase offset；domain-time shear 表示这些相位差是否随 bridge time 发生不同变化。二者分别对应域校准与时间条件校准。

这为后续算法提供了呼应关系，但本轮仍未证明 `G_t*` 会导致最终恢复质量下降，也未证明 DT/HJ 能降低它。

## 6. 工程与身份

- 训练状态：`{training['status']}`；耗时 {training['elapsed_seconds'] / 60:.1f} 分钟。
- 测量状态：`{measurement['status']}`；原始 age rows {measurement['age_rows']}，primary rows {measurement['primary_rows']}。
- 目标像素进入测量：`{measurement['target_content_read']}`。
- 训练使用配对关系：`{training['pairing_used_for_training']}`。
- 数据 manifest：`{prepare['data_manifest_sha256']}`。
- held-out manifest：`{prepare['heldout_manifest_sha256']}`。
- 主原始证据：`raw/RECIPROCAL_KERNEL_BY_AGE.csv`。
- 统计裁决：`reports/PHASE_STATISTICS.json`。
- 头图：`figures/UNSB_SIXDOMAIN_PHASE_HEADFIGURE.pdf`。

{previous_section}
## 8. 不能越过的边界

- 只有一个新训练 seed=2051；图像 bootstrap 不能替代多训练 seed。
- held-out 来自同一套本地数据源，不是 external/sealed confirmation。
- RainDS 的 200 个身份已被 120 trainA + 80 held-outA 用尽，本地没有剩余同分布确认身份。
- KDD 比较的是条件平均 endpoint direction，不是完整 Schrödinger transition kernel 的严格概率度量。
- 正的 shared-clock regret 是过程几何错配，不等于因果恢复伤害。
- 本轮没有候选算法，因此不能声称 phase correction、DT 或 HJ 已被该头图验证。
"""
    output = root / "reports" / "UNSB_SIXDOMAIN_PHASE_FINAL_CN.md"
    output.write_text(report, encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
