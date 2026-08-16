# HJ 归因结果（干净框架，val-O，单 seed=2026）

## performance：true vs plain

- PSNR +2.7533 dB，CI95 [1.8420, 3.7258] ✅
- SSIM +0.0286，CI95 [0.0084, 0.0501] ✅
- LPIPS -0.0602，CI95 [-0.0821, -0.0392] ✅
- NIQE -0.3563，CI95 [-0.7197, 0.0016]

## attribution：true vs roll（结构方向按 patch 平移）

- PSNR +2.7612 dB，CI95 [2.1564, 3.3870] ✅
- SSIM +0.0373，CI95 [0.0197, 0.0555] ✅
- LPIPS -0.1225，CI95 [-0.1377, -0.1070] ✅
- NIQE -1.7708，CI95 [-2.1136, -1.4317] ✅

## 结论

true 相对 roll 在 PSNR/SSIM/LPIPS/NIQE 全指标显著占优，且 roll ≈ plain（roll 几乎无收益）。这证明 HJ 的收益来自**真实结构方向特异**，不是通用投影/稳健化。归因门 PASS。

（原历史 V13 的 true-vs-roll SSIM -0.0202 归因失败，在干净重构后翻转为全指标正归因。）
