# README_reproduce

Run from `multi_agent_shared_write_named/`.

```bash
BUNDLED_PY=/Users/changyitong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s MASW/tests
PYTHONDONTWRITEBYTECODE=1 "$BUNDLED_PY" -m experiments.run_real_experiments
PYTHONDONTWRITEBYTECODE=1 "$BUNDLED_PY" experiments/update_document_chapter.py
python3 /Users/changyitong/.codex/plugins/cache/openai-bundled/latex/0.2.4/scripts/compile_latex.py "$(pwd)/document.tex" --engine xelatex
```

Outputs:

- `artifacts/experiment_audit.md`
- `artifacts/experiment_plan.md`
- `artifacts/benchmark_protocol.md`
- `results/`
- `figures/`
- `data/`
- `document.tex`
- `document.pdf` after LaTeX compilation
