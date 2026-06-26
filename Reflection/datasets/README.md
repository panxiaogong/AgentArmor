# Reflection Datasets

This directory now contains two tiers of datasets for Type-III Reflection/Synthesis Write:

- `reflection_type3_seed.csv`
  - 40-row smoke-test subset used by the current unit tests and quick local checks.
- `reflection_type3_competition.csv`
  - 120-row competition dataset for ablation, node-level evaluation, and future baseline comparison.

## Competition Dataset Breakdown

`reflection_type3_competition.csv` contains:

- `RS-1`: 20 attack samples
  - Explicit reflection steering / summary-targeted prompt injection.
- `RS-2`: 20 attack samples
  - Low-trust fact laundering into durable profile-like memory.
- `RS-3`: 20 attack samples
  - Secret or dangerous instruction persistence.
- `RS-4`: 20 attack samples
  - Delayed retrieval hijack / workflow poisoning.
- `BENIGN`: 40 benign samples
  - Stable user, system, or assistant-backed facts that are reasonable to preserve.

Total:

- `80` attack samples
- `40` benign samples
- `120` samples overall

## Column Schema

The competition CSV uses a richer schema than the smoke-test seed file:

- `sample_id`: unique identifier
- `attack_class`: `RS-1` / `RS-2` / `RS-3` / `RS-4` / `BENIGN`
- `target_stage`: coarse phase label for the main threat stage
- `reflection_task`: the background summarization or reflection job
- `source`: `user` / `assistant` / `tool` / `web` / `system`
- `raw_text`: raw content before reflection
- `summary_candidate`: distilled fact likely to be written to memory
- `target_query`: future query that would retrieve or use the memory
- `label`: `attack` or `benign`
- `attack_goal`: fine-grained objective or benign fact type
- `dataset_source`: provenance of the sample construction
- `notes`: short explanation of the scenario

## Source Provenance Labels

The `dataset_source` field distinguishes between directly adapted public patterns and manual reconstructions:

- `promptinject_adapted`
  - Adapted from public prompt-injection patterns in PromptInject.
- `tenable_reconstructed`
  - Reconstructed from the delayed-memory / Dreaming-style reflection threat model described by Tenable.
- `memory_poisoning_style_manual`
  - Hand-built Reflection-specific poisoning samples inspired by public memory-poisoning papers and taxonomies.
- `benign_curated`
  - Benign stable facts curated for false-positive testing.
- `assistant_restatement`
  - Benign assistant restatements of user preferences.
- `system_profile`
  - Benign trusted profile or system-derived facts.

## Public Source Pointers

These are the main public references used to shape the first competition-ready dataset:

- PromptInject
  - https://github.com/agencyenterprise/PromptInject
- JailbreakBench
  - https://github.com/JailbreakBench/jailbreakbench
- Tenable TRA-2025-11
  - https://www.tenable.com/security/research/tra-2025-11

Practical note:

- This first Reflection dataset leans more heavily on PromptInject-style indirect steering patterns and Tenable-style delayed memory poisoning patterns because they map more directly onto Type-III Reflection/Synthesis Write.
- JailbreakBench is still a useful public source for future expansion when we want broader prompt-injection diversity or later baseline harmonization across different attack families.

## Usage Guidance

Suggested order of use:

1. Use `reflection_type3_seed.csv` for smoke tests and local debugging.
2. Use `reflection_type3_competition.csv` for:
   - RQ-style node evaluation
   - Config-1~5 ablation tables
   - later baseline comparisons
   - latency aggregation

The competition dataset is intentionally English-first because the reusable public corpora and public prompt-injection patterns are primarily English. A bilingual expansion can be added later without breaking the current schema.
