# MASW: Multi-Agent Shared Write

这个目录采用和参考图一致的命名方式：`config.py`、`types.py`、`pipeline.py`
加 `d1_*.py` 到 `d7_*.py` 的防御节点文件。

## 文件说明

- `d1_input_label.py`: 外部输入低信任标记。
- `d2_candidate_extract.py`: Agent A 候选事实抽取。
- `d3_risk_filter.py`: 注入与指令型载荷风险检测。
- `d4_provenance_gate.py`: provenance、验证提升、共享记忆写入网关。
- `d5_retrieval_audit.py`: 信任感知检索与审计。
- `d6_execution_align.py`: 动作提案与工具调用仲裁。
- `d7_revocation.py`: 污染记忆追踪与撤销。

## 运行

从 `Multi-Agent Shared Write` 的父目录或当前目录执行：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s MASW/tests
PYTHONDONTWRITEBYTECODE=1 python3 -m MASW.tests.eval_masw
```
