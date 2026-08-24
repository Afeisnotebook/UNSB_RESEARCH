# 域相位第一次确认协议的 null 失效记录

`unsb-shared-bridge-domain-phase-desynchronization-v1` 的原始数值输出永久保留，但其 domain-label permutation 裁决记为 `INVALID_NULL_EXCHANGEABILITY`，不是科学 FAIL。

原因：一个域内的五年龄 KDD profile 是“AIO 相对该域五个 Single checkpoint”的结果。不同域的 profile 由不同参考模型系统生成，绝对 KDD 量纲、曲线形状和年龄谷底都不具备跨域交换性。原 null 把完整 image profile 随机分到其它域标签后求组均值，产生的是混合参考模型系统的年龄谷底；它没有检验预先固定的“域身份→有效年龄”映射能否在新图像上复现。

失效协议仍提供以下描述事实：

- 24 张/域的第一内部确认 split 与 20 张/域发现 split 在 15/15 个域–时单元上给出完全相同的有效年龄图；
- M16 与 M32 在 15/15 单元一致；
- 14/15 单元的 bootstrap modal share 不低于 80%。

这些事实用于形成新的固定预测，但不能直接充当修复后的确认检验。修复协议另取第三个零重叠 split，只在每个 bridge time 内打乱预先固定的五个年龄指派，不移动任何实测 KDD profile 或参考模型系统。
