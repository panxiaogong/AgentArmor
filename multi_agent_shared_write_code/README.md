# Multi-Agent Shared Write Defense

这是一个面向“多 Agent 共享写入”场景的防御原型。它把共享记忆系统拆成多个可审计的防御节点，避免低信任外部输入通过 Agent 写入共享记忆后被高权限 Agent 当作可信指令执行。

核心安全原则：

```text
No automatic trust promotion.

任何外部内容不能因为“被某个 Agent 写入共享记忆”
就自动从低信任升级为高信任。
```

## 目录结构

```text
shared_write_defense/
  types.py              # 共享数据结构与信任级别
  input_ingestion.py    # 外部输入标记与 spotlighting
  risk.py               # 注入风险、指令相似度、上下文风险检测
  extraction.py         # Agent A 从外部内容抽取候选事实
  memory_store.py       # 内存版共享记忆、隔离区、审计日志
  conflict_detection.py # 事实冲突检测与置信度计算
  memory_gateway.py     # 共享记忆写入网关
  verification.py       # 候选事实验证与信任提升
  retrieval.py          # 信任感知检索
  action_mediator.py    # 工具调用仲裁器
  revocation.py         # 污染记忆追踪与撤销
  pipeline.py           # 将各防御节点编排为完整拦截路径
examples/
  demo.py               # 最小可运行演示
tests/
  test_pipeline.py      # 基础安全不变量测试
```

## 运行演示

在本目录下执行：

```bash
python -m examples.demo
```

运行测试：

```bash
python -m unittest discover -s tests
```

## 设计说明

本实现没有绑定具体 LLM 或向量数据库，所有外部模型调用都通过清晰的接口位置预留。这样便于后续替换为 AutoGen、CrewAI、LangGraph、Mem0 或自研 Agent 框架。

当前版本采用纯 Python 标准库，便于直接阅读、测试和改造。
