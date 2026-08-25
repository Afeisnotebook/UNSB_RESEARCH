# 关于 `code/` 与 `dtcov/`

实际可导入的 Python 包是 `dtcov/`。这里没有把包命名为 `code`，因为
`code` 是 Python 标准库模块名；如果把它作为包放到 `sys.path`，会破坏
`pdb`/`torch.distributed` 的内部 `import code`。

因此：

- 核心算法：`dtcov/dtcovmatch.py`
- 训练接入：`dtcov/model.py`
- 单元测试：`tests/test_dtcovmatch.py`

本目录仅保留这个说明，避免误把目录名当包名。
