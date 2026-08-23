# clean_reexploration 模块

本目录实现 `4090_DTHJ_HNEK_CLEAN_REEXPLORATION_LONG_TASK_PROMPT_CN_20260824.md`
定义的 DT / HJ / HNEK 干净工程再探索。所有新逻辑集中在本目录，只对 baseline 做
最小方法注册接入（`set_train_epoch`、HNEK ON/OFF 开关）。

## 文件

- `identity.py`：基座权威、数据/代码/run 身份与 SHA-256。
- `controllers.py`：统一目标盲控制器、聚类 bootstrap 与冻结裁决规则。
- `diagnostics.py`：目标盲诊断面板选择与纯统计量。
- `full_state.py`：确定性 full-state 捕获、保存、加载与恢复。
- `access_guard.py`：paired target 访问闸门与只追加 access ledger。
- `adjudicate.py`：机械裁决（纯函数）。
- `package_return.py`：唯一 ZIP 与外部 SHA-256 sidecar。
- `run_long.py`：长任务编排入口。
- `train_executor.py`：确定性训练循环与 lane 执行。

## 执行

```bash
bash LAUNCH_4090_CLEAN_REEXPLORATION.sh
```

普通进度只写日志和 heartbeat，不触发作者闸门。任务完全结束或全局 hard stop 后才
生成唯一 ZIP 与 sidecar。
