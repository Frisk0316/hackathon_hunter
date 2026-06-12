# Implementation Phases

## Phase 1 — Minimal Viable Harness

新增：

```text
AI_CONTEXT.md
AGENTS.md
CLAUDE.md
docs/ARCHITECTURE.md
docs/FEATURE_MAP.md
docs/RUNBOOK.md
Makefile
```

完成後應能回答：這個專案是什麼、功能在哪裡、怎麼跑、怎麼測、AI 改 code 的規則是什麼。

## Phase 2 — Doc Sync Harness

新增：

```text
docs/DOMAIN_RULES.md
docs/DOC_IMPACT_MATRIX.md
docs/CHANGE_MANIFEST_TEMPLATE.md
docs/ADR/README.md
scripts/docs/check_doc_impact.py
Makefile target: docs-impact
```

完成後應能回答：改了某個檔案，要檢查哪些文件？業務規則變了，要不要新增 ADR？

## Phase 3 — Intelligence Harness

優先新增：

```text
docs/MENTAL_MODELS.md
docs/INVARIANTS.md
docs/FAILURE_MODES.md
```

研究型專案再新增：

```text
docs/HYPOTHESIS_LEDGER.md
docs/EXPERIMENT_REGISTRY.md
docs/GOLDEN_CASES.md
docs/CRITIQUE_PROTOCOL.md
docs/QUESTION_BANK.md
```

## Phase 4 — Context Resilience Harness

新增：

```text
docs/CURRENT_STATE.md
docs/CONTEXT_INDEX.md
docs/CONTEXT_BUDGET.md
docs/COMPRESSION_RULES.md
docs/CONTEXT_PACKS/
tasks/CONTEXT_HANDOFF_TEMPLATE.md
```
