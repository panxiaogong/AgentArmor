# Dataset Sources And Construction Notes

This minimum dataset is synthetic and deterministic. It is designed as a
drop-in first pass before importing larger external benchmarks.

Reference sources reviewed for future replacement or expansion:

- JailbreakBench: https://github.com/JailbreakBench/jailbreakbench
- PromptInject / Ignore Previous Prompt style attacks: https://github.com/agencyenterprise/PromptInject
- AgentDojo agent prompt-injection benchmark: https://github.com/ethz-spylab/agentdojo

Why synthetic first:

- The MASW prototype needs structured labels for `prompt_injection`,
  `tool_misuse`, `memory_poisoning`, and `agent_hijacking`.
- Public datasets often focus on jailbreak or indirect injection text, but do
  not always include MASW-specific expected outcomes such as memory write,
  quarantine, or action-mediation decisions.
- This file keeps the schema stable so benchmark rows can be imported later
  without changing tests.

Minimum counts:

- prompt_injection: 20
- tool_misuse: 20
- memory_poisoning: 20
- agent_hijacking: 20
- benign: 45
