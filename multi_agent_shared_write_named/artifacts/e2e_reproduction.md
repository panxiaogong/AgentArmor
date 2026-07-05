# E2E Reproduction

The E2E experiment uses a real SQLite memory database and real sandboxed tools.

```bash
BUNDLED_PY=/Users/changyitong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3
PYTHONDONTWRITEBYTECODE=1 "$BUNDLED_PY" -m experiments.run_real_experiments
```

Relevant outputs:

- `data/type5/e2e_cases.jsonl`: fixed cases and labels.
- `results/e2e/e2e_memory.sqlite`: persistent memory backend.
- `results/e2e/tool_sandbox/tools.sqlite`: sandbox tool database.
- `results/e2e/tool_sandbox/tool_calls.jsonl`: executed sandbox tool calls.
- `results/e2e/raw_logs/*.json`: per-case audit logs.
- `results/e2e/summary.csv`: aggregate metrics.
- `results/e2e/per_case.jsonl`: per-case outcomes.

No real email, production deploy, secret read, or destructive command is executed.
