# MASW Technical Selection Report

## Decision

- Recommended configuration: `D3_HYBRID_PLUS_D4_RULE`
- Keep `d4_provenance_gate` as the non-negotiable architecture boundary.
- Use `HybridDetector` in `d3` for early screening.
- Keep `RuleBasedDetector` inside `d4` verifier for now because the lower verification threshold makes hybrid more likely to false-positive.
- `RubricDetector` is a deterministic stand-in for an LLM judge and should be replaced by a real model only after privacy/cost tests.

## Metrics Summary

| Config | Precision | Recall | F1 | FPR |
|---|---:|---:|---:|---:|
| `D3_RULE` | 1.0000 | 0.8889 | 0.9412 | 0.0000 |
| `D3_RUBRIC` | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| `D3_HYBRID` | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| `D4_GATE_NO_DETECTOR` | 1.0000 | 0.8889 | 0.9412 | 0.0000 |
| `D4_GATE_RULE` | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| `D4_GATE_HYBRID` | 0.9783 | 1.0000 | 0.9890 | 0.0364 |
| `D3_RULE_PLUS_D4_RULE` | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| `D3_HYBRID_PLUS_D4_RULE` | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| `D3_HYBRID_PLUS_D4_HYBRID` | 0.9783 | 1.0000 | 0.9890 | 0.0364 |
| `ALL_RECOMMENDED` | 1.0000 | 1.0000 | 1.0000 | 0.0000 |

## Why This Choice

- Best observed F1: 1.0000; tied configs: `D3_RUBRIC`, `D3_HYBRID`, `D4_GATE_RULE`, `D3_RULE_PLUS_D4_RULE`, `D3_HYBRID_PLUS_D4_RULE`, `ALL_RECOMMENDED`.
- `D3_RULE` recall = 0.8889; it is fast and auditable but brittle on subtle sync/exfiltration phrasing.
- `D3_HYBRID` recall = 1.0000; it keeps rule precision and adds semantic coverage.
- `D4_GATE_NO_DETECTOR` recall = 0.8889; pure provenance without a detector is insufficient.
- `D4_GATE_HYBRID` FPR = 0.0364; hybrid is too aggressive inside the stricter verifier threshold.
- `D3_HYBRID_PLUS_D4_RULE` F1 = 1.0000.
- `D3_RUBRIC` ties on this small set, but the recommended deployment keeps the rule detector in the loop so known high-confidence patterns remain explicit and auditable.

Conclusion: choose `D3 HybridDetector + D4 RuleBasedDetector`. This is not because regex alone failed on the first easy set; it is because the second hard set contains stealthy exfiltration/sync wording where semantic early screening improves recall, while D4 should remain conservative to protect benign internal sync flows.
