# AI-Native Development Harness Overview

## Problem

在 GPT、Codex、Claude Code 的協助下，專案開發速度會大幅提升，但也帶來新的風險：

- 程式碼增長速度超過人類理解速度。
- 小 bug 不知道去哪裡改，只能再丟 prompt。
- AI session 有限制，工作節奏被工具 quota 牽制。
- 一旦 AI 停機或上下文壓縮，人類難以接手。
- 文件若不更新，AI 會讀到舊 context，進一步放大錯誤。
- AI 可能只是執行人類想法，而沒有幫人類擴展思考。

## Solution

完整架構分成五層：

```text
Control Harness       讓 AI 不失控
Navigation Harness    讓人類知道去哪裡改
Doc Sync Harness      讓文件跟著業務邏輯同步
Intelligence Harness  讓人類累積理解與判斷力
Context Resilience    讓長對話、壓縮、換 session、換模型時不失憶
```

## 最小可行架構

```text
AI_CONTEXT.md
AGENTS.md
CLAUDE.md
docs/ARCHITECTURE.md
docs/FEATURE_MAP.md
docs/RUNBOOK.md
Makefile
```

## 不要過度文件化

只有當 repo 出現對應痛點時，才補對應文件：

| 痛點 | 補充文件 |
|---|---|
| 不知道功能在哪裡 | FEATURE_MAP.md |
| UI 常常找不到檔案 | UI_MAP.md |
| API contract 不清楚 | API_MAP.md |
| 資料流很複雜 | DATA_FLOW.md |
| 業務邏輯常變 | DOMAIN_RULES.md, DOC_IMPACT_MATRIX.md |
| AI 常破壞核心假設 | INVARIANTS.md |
| 常重複踩同一種 bug | FAILURE_MODES.md |
| 策略研究實驗很多 | HYPOTHESIS_LEDGER.md, EXPERIMENT_REGISTRY.md |
| 對話太長、session 常斷 | CURRENT_STATE.md, CONTEXT_INDEX.md |
| AI 太順從 | CRITIQUE_PROTOCOL.md, DESIGN_SPACE.md |
