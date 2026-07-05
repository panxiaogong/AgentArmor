# AgentArmor Experiment Plan

## Must Complete

| Experiment | Research question | Comparator | Data | Metrics | Command | Output | External dependency | Success/Failure |
|---|---|---|---|---|---|---|---|---|
| SQLite E2E | Does AgentArmor block attack input -> memory -> retrieval -> tool execution? | baseline vs protected | 30 attack + 8 benign | write/retrieval/plan/tool rates, latency | `PYTHONDONTWRITEBYTECODE=1 python3 -m experiments.run_real_experiments` | `results/e2e/` | no | success if raw logs and summary are generated |
| Type4 full system | Does a full Type4 local pipeline beat simple baselines on the same data? | keyword/perplexity/AgentArmor/ablations | main + hard set | Precision/Recall/F1/FPR/AUPRC/AUROC/latency | same | `results/type4/` | no for local; DeepSeek blocked without key | success if CSVs generated and blocked remote noted |
| Type5 hard set | Does hard set expose D3/D4 saturation and downstream risks? | no defense, D3, D4, D3+D4, complete | Type5 hard set | classification + propagation/execution/revocation | same | `results/type5/` | no | success if hardset/e2e/revocation CSVs exist |
| Leave-one-out | Which D node matters in complete config? | ALL, ALL-D1..D7, D3+D4, none | Type5 hard set, 3 seeds | mean/std/CI | same | `results/ablation/` | no | success if repeated rows generated |
| Unified benchmark | Are latency figures comparable? | local methods + blocked API entries | all generated outputs | P50/P95/P99/throughput/resource | same | `results/performance/` | no | success if protocol and CSVs exist |
| Benign FPR | What benign content is falsely blocked? | AgentArmor local detector | 216 benign | FP categories | same | `results/benign/` | no | success if corpus and FP CSV exist |

## Try If Environment Allows

- Real LangChain/LlamaIndex/AutoGen/CrewAI integration.
- Real Chroma/FAISS/Qdrant vector database.
- DeepSeek remote LLM-only baseline.

## Environment-Limited Alternative

If external packages, network, GPU, or API keys are unavailable, use the local SQLite memory backend and sandbox tools, and report the external framework/API experiments as blocked. Do not report substitute numbers as remote framework or remote LLM results.
