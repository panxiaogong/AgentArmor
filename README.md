# Multi-Agent Shared Write

源码放在 `MASW/` 包内。

该模块实现多 Agent 共享写入场景下的防御原型，包括：

- 外部输入低信任标记
- 候选事实抽取
- 注入风险过滤
- provenance gate
- 信任感知检索
- 工具调用仲裁
- 污染记忆撤销

## Run

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s MASW/tests
PYTHONDONTWRITEBYTECODE=1 python3 -m MASW.tests.eval_masw
