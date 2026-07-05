# MASW Baseline Comparison Report

## Dataset

- Total samples: 145
- Source: `build_selection_samples()` = MASW minimum dataset plus subtle sync/exfiltration hard set.
- Labels: attack vs benign.

## Baselines

- `MASW_ALL_RECOMMENDED`: our system, using D3 HybridDetector + D4 RuleBasedDetector + D5/D6.
- `DEEPSEEK_V4_FLASH`: remote LLM-as-judge baseline. It is skipped unless local env enables it.
- `PROMPTINJECT_STYLE_TOOL`: prompt injection keyword guard.
- `LLM_GUARD_STYLE_TOOL`: local multi-scanner guard approximation.
- `REBUFF_STYLE_TOOL`: local layered signature/canary-style guard approximation.

## Metrics

| Baseline | Status | Precision | Recall | F1 | FPR | P95 ms |
|---|---|---:|---:|---:|---:|---:|
| `MASW_ALL_RECOMMENDED` | ok | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.1124 |
| `DEEPSEEK_V4_FLASH` | skipped | skipped | skipped | skipped | skipped | skipped |
| `PROMPTINJECT_STYLE_TOOL` | ok | 1.0000 | 0.3222 | 0.4874 | 0.0000 | 0.0095 |
| `LLM_GUARD_STYLE_TOOL` | ok | 1.0000 | 0.4333 | 0.6047 | 0.0000 | 0.0092 |
| `REBUFF_STYLE_TOOL` | ok | 1.0000 | 0.4222 | 0.5938 | 0.0000 | 0.0084 |

## DeepSeek Run Note

DeepSeek was not executed in this run. To enable it locally, create `.env` from `.env.example`, set `DEEPSEEK_API_KEY`, and set `MASW_RUN_REMOTE_BASELINES=1`.

## Interpretation

The local tool baselines are deterministic adapters for comparison. They are intentionally weaker than a full MASW pipeline because they only classify text; they do not enforce provenance, trust promotion, retrieval filtering, or action mediation.
