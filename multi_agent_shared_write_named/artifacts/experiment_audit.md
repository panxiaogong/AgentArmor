# AgentArmor Experiment Audit

Generated at: 2026-07-03T03:27:25.045608+00:00

## Project Structure

- `MASW/`: current workspace implementation for Type 5 multi-agent shared write, including D1-D7 modules, tests, reports, and prior results.
- `multi_agent_shared_write_code/`: smaller prototype with similar shared-write defense components.
- Uploaded `AgentArmor-main-2.zip`: audited in `/private/tmp/agentarmor_zip_audit` and found to contain Type 1 AutoWrite, Type 2 MINJA, Type 3 Reflection, Type 4 External_Developer_Write, and Type 5 MASW code. It was not copied over the working tree.
- `document.tex`: copied into the project root from the WeChat temp path for reproducible editing and compilation.

## Existing Tests And Results

- Existing MASW unit tests and evaluators: `MASW/tests/*.py`.
- Existing MASW dataset: `MASW/tests/data/masw_min_dataset.jsonl`, 80 attack + 45 benign.
- Existing MASW results: `MASW/tests/results/*.csv`, `*.json`, `*.md`.
- Existing PDF reviewed visually through PDF tooling; text extraction is unreliable because of Chinese font encoding.

## Five Test Scenarios

| Type | Scenario | Current status before this run |
|---|---|---|
| Type 1 | Framework automatic memory write / long-term memory pollution | Present in uploaded zip (`AutoWrite`), not integrated in current workspace; old chapter numbers were not traceable here. |
| Type 2 | Agent-initiated write / MINJA-style autonomous memory write | Present in uploaded zip (`MINJA`), not integrated in current workspace; old chapter numbers were not rerun here. |
| Type 3 | Reflection, summary, or memory update pollution | Present in uploaded zip (`Reflection`), not integrated in current workspace; old DeepSeek latency numbers are external to this run. |
| Type 4 | Supply-chain / knowledge-base / document pollution | New complete local experiment added in `results/type4/`. Remote LLM baseline blocked without key. |
| Type 5 | Multi-agent shared memory and permission/tool misuse | New hard set, E2E SQLite chain, revocation, leave-one-out, and benign analyses added. |

## Data, Labels, Thresholds, Leakage Risk

- New Type 4 data is deterministic manual/synthetic data saved in `data/type4/`. Labels are assigned by construction and preserved per row.
- New Type 5 hard set is deterministic manual/synthetic data saved in `data/type5/hardset.jsonl`.
- E2E data is saved in `data/type5/e2e_cases.jsonl`; it contains 30 malicious attack chains and 8 benign chains.
- Benign corpus has 216 rows saved in `data/benign_memory_corpus/`.
- Thresholds are fixed in code (`RISK_THRESHOLD_WRITE=0.45` or documented local thresholds) before evaluation. No threshold is tuned on test output.
- Leakage risk: all new synthetic datasets are generated and evaluated by fixed rules, so results should be interpreted as engineering regression and hard-set evidence, not independent public benchmark generalization.

## Latency Metric Audit

The previous chapter mixed at least two measurement scopes:

- Local deterministic MASW or rule/rubric paths measured only Python in-process execution, often sub-millisecond.
- DeepSeek/API rows measured remote model inference and network time in other scripts, often seconds.

These are not comparable unless tables explicitly state whether network, model inference, embedding, retrieval, and logging are included. The new `results/performance/unified_latency.csv` records these fields.

## External Experiment Blockers

- `external_framework_langchain`: blocked Python package import failed: ModuleNotFoundError; network is restricted so dependency install was not attempted
- `external_framework_llama_index`: blocked Python package import failed: ModuleNotFoundError; network is restricted so dependency install was not attempted
- `external_framework_autogen`: blocked Python package import failed: ModuleNotFoundError; network is restricted so dependency install was not attempted
- `external_framework_crewai`: blocked Python package import failed: ModuleNotFoundError; network is restricted so dependency install was not attempted
- `external_framework_chromadb`: blocked Python package import failed: ModuleNotFoundError; network is restricted so dependency install was not attempted
- `external_framework_faiss`: blocked Python package import failed: ModuleNotFoundError; network is restricted so dependency install was not attempted
- `external_framework_qdrant_client`: blocked Python package import failed: ModuleNotFoundError; network is restricted so dependency install was not attempted
- `DeepSeek remote LLM baseline`: blocked DEEPSEEK_API_KEY and MASW_RUN_REMOTE_BASELINES=1 were not both present; remote API call not executed
