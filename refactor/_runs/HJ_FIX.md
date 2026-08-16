# HJ 集成 bug 根因与修复

## 根因

`probe_fn` 里用 `netG(..., [layer], encode_only=True)` 只传单个 layer。原始 `ResnetGenerator_ncsn.forward` 的 `encode_only` 提前返回条件是 `layer_id + len(self.model) == layers[-1]`，当 `layers=[0]` 时 `layers[-1]=0` 永远不满足，于是返回 `(feat, feats)` 元组而非特征列表，导致 netF 拿到错误特征形状，MLP 报 `mat1 (256x256) vs mat2 (3x256)`。

## 修复

`probe_fn` 改为用完整 `self.nce_layers` 调 netG，再按 `self.nce_layers.index(layer)` 取对应 pooled feature。

## 验证

HJ GPU smoke（4 图 1 epoch）通过，无 crash。
