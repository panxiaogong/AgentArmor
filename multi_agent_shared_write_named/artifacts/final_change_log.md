# Final Change Log

## New Experiments

- Added SQLite-backed E2E baseline/protected attack-chain experiment.
- Added Type4 full-system and hard-set comparison.
- Added Type5 hard set, multi-agent E2E metrics, and revocation test.
- Added complete leave-one-out ablation with 3 repeated seeds.
- Added unified latency/resource protocol and CSV outputs.
- Added 216-row benign memory corpus and false-positive analysis.

## Retained Old Results

- Existing MASW test outputs remain in `MASW/tests/results/` for traceability.

## Replaced Or Removed From Report

- Untraceable old Type1/Type2/Type3/Type4/DeepSeek numbers are not reused as new evidence in the rewritten Chapter 3.
- Remote LLM/API rows are reported as blocked unless actually executed.

## New Tables And Figures Source Mapping

- E2E tables/figures: `results/e2e/summary.csv`, `results/e2e/per_case.jsonl`.
- Type4 tables/figures: `results/type4/full_system_main.csv`, `results/type4/hardset.csv`, `results/type4/error_analysis.csv`.
- Type5 tables/figures: `results/type5/hardset_main.csv`, `results/type5/e2e_mult_agent.csv`, `results/type5/revocation.csv`.
- Ablation figures: `results/ablation/leave_one_out.csv`.
- Performance figures: `results/performance/unified_latency.csv`.
- Benign figures: `results/benign/false_positive_analysis.csv`.

## Incomplete Experiments And Real Reasons

- `external_framework_langchain`: blocked Python package import failed: ModuleNotFoundError; network is restricted so dependency install was not attempted
- `external_framework_llama_index`: blocked Python package import failed: ModuleNotFoundError; network is restricted so dependency install was not attempted
- `external_framework_autogen`: blocked Python package import failed: ModuleNotFoundError; network is restricted so dependency install was not attempted
- `external_framework_crewai`: blocked Python package import failed: ModuleNotFoundError; network is restricted so dependency install was not attempted
- `external_framework_chromadb`: blocked Python package import failed: ModuleNotFoundError; network is restricted so dependency install was not attempted
- `external_framework_faiss`: blocked Python package import failed: ModuleNotFoundError; network is restricted so dependency install was not attempted
- `external_framework_qdrant_client`: blocked Python package import failed: ModuleNotFoundError; network is restricted so dependency install was not attempted
- `DeepSeek remote LLM baseline`: blocked DEEPSEEK_API_KEY and MASW_RUN_REMOTE_BASELINES=1 were not both present; remote API call not executed
