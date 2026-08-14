# AgentArmor Demo Web

这是 AgentArmor 的本地真实模块演示页。浏览器负责展示，Python 服务会为每次运行创建新的证据目录，并实际调用项目防御模块。

## 启动

在本目录运行：

```powershell
python demo_server.py
```

然后打开 `http://127.0.0.1:4173`。

## 当前范围

- 无防护与 AgentArmor 双轨对比
- 逐步执行、自动播放和重置
- 端到端污染、知识库后门、多 Agent、级联撤销和完整性场景
- 记忆数据库、审计日志和实验指标面板

页面不使用模拟结果兜底。点击“执行下一步”或“自动播放”后，会请求 `/api/run`，只有后端返回 `real_execution=true` 才展示结果；模块异常会直接显示为运行失败。

每次运行的 SQLite、工具沙箱与完整性证据保存在 `runtime/REAL-*`，该目录已被 Git 忽略。
