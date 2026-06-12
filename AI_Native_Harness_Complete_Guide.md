---

<!-- FILE: README.md -->

# AI-Native Repo Harness Pack

Last updated: 2026-06-12

這是一套可套用到不同 repository 的 AI-native 開發治理文件包，目標是避免長對話、上下文壓縮、AI session limit、工具切換或模型切換造成專案失憶。

它不是要求每個 repo 都建立大量文件，而是提供一個「漸進式啟用」的工具箱：

1. **Control Harness**：讓 AI 不失控。
2. **Navigation Harness**：讓人類知道功能、檔案、資料流在哪裡。
3. **Doc Sync Harness**：讓業務邏輯改變時，文件能同步更新。
4. **Intelligence Harness**：讓人類能從 AI 產出中累積理解與判斷力。
5. **Context Resilience Harness**：讓長 session、上下文壓縮、換模型、換工具時不失憶。

## 快速使用

先讀：

```text
00_PLAYBOOK/01_HARNESS_LEVELS.md
```

大多數 repo 先從最小可行版本開始：

```text
AI_CONTEXT.md
AGENTS.md
CLAUDE.md
docs/ARCHITECTURE.md
docs/FEATURE_MAP.md
docs/RUNBOOK.md
Makefile
```

如果 repo 有複雜業務邏輯，再啟用 Doc Sync。若希望人類真正理解專案，再啟用 Intelligence Harness。若遇到長對話或 session 斷裂，再啟用 Context Resilience。

> 文件不是越多越好，而是越能降低理解成本越好。當文件開始比它解釋的系統還複雜，harness 就已經失敗。


---

<!-- FILE: AI_NATIVE_HARNESS_MASTER_GUIDE.md -->

# AI-Native Harness Master Guide

這份文件是整套文件包的壓縮總覽。之後如果上下文被壓縮，可以先讀這份，再依需要展開其他模板。

## 1. 核心問題

AI 能快速生成專案，但會造成：

- 人類對架構理解追不上 code growth。
- 文件老化，AI 讀到舊 context。
- 長對話、session limit、上下文壓縮造成資訊遺失。
- AI 只執行人類想法，沒有提升人類思考能力。

## 2. 核心解法

把 repo 設計成外部記憶系統：

```text
Control Harness
Navigation Harness
Doc Sync Harness
Intelligence Harness
Context Resilience Harness
```

## 3. 最小可行 Harness

```text
AI_CONTEXT.md
AGENTS.md
CLAUDE.md
docs/ARCHITECTURE.md
docs/FEATURE_MAP.md
docs/RUNBOOK.md
Makefile
```

## 4. 什麼時候加文件？

| 痛點 | 文件 |
|---|---|
| 功能不知道在哪 | FEATURE_MAP.md |
| UI 不知道在哪改 | UI_MAP.md |
| API 契約不清楚 | API_MAP.md |
| 資料流複雜 | DATA_FLOW.md |
| 業務規則常變 | DOMAIN_RULES.md, DOC_IMPACT_MATRIX.md |
| AI 常破壞核心假設 | INVARIANTS.md |
| 常重複踩 bug | FAILURE_MODES.md |
| 研究實驗很多 | HYPOTHESIS_LEDGER.md, EXPERIMENT_REGISTRY.md |
| 對話太長或 session 常斷 | CURRENT_STATE.md, CONTEXT_INDEX.md |
| AI 太順從 | CRITIQUE_PROTOCOL.md, DESIGN_SPACE.md |

## 5. AI 任務標準流程

```text
Read context
-> git status
-> locate before edit
-> design-space expansion if non-trivial
-> minimal implementation
-> tests
-> doc impact check
-> handoff
-> human learning notes
-> context handoff
```

## 6. 上下文限制原則

不要把 chat history 當 source of truth。repo docs、git diff、tests 才是專案事實來源。

壓縮時不能省略：

```text
active task
do-not-touch files
business logic changes
failed tests
docs impacted
human decisions
rollback plan
uncertainty
```


---

<!-- FILE: 00_PLAYBOOK/00_OVERVIEW.md -->

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


---

<!-- FILE: 00_PLAYBOOK/01_HARNESS_LEVELS.md -->

# Harness Levels

不要對所有 repo 套同一套完整文件。先依據複雜度選擇等級。

## Level 0 — 一次性腳本 / Demo

適合小工具、一次性資料處理、短期 demo。

建議文件：

```text
README.md
AGENTS.md
docs/RUNBOOK.md
```

不要加入 Intelligence Harness、ADR、Doc Sync。

## Level 1 — 小型 AI-Assisted 專案

適合小型 Telegram bot、簡單前後端、小型研究工具、單人維護 project。

```text
AI_CONTEXT.md
AGENTS.md
CLAUDE.md
docs/ARCHITECTURE.md
docs/FEATURE_MAP.md
docs/RUNBOOK.md
docs/CHANGELOG_AI.md
Makefile
```

## Level 2 — 中型長期專案

適合量化回測平台、資料 pipeline、前後端 dashboard、持續演進的 app。

```text
AI_CONTEXT.md
AGENTS.md
CLAUDE.md
docs/ARCHITECTURE.md
docs/FEATURE_MAP.md
docs/UI_MAP.md
docs/API_MAP.md
docs/DATA_FLOW.md
docs/DOMAIN_RULES.md
docs/DOC_IMPACT_MATRIX.md
docs/RUNBOOK.md
docs/CHANGELOG_AI.md
docs/KNOWN_ISSUES.md
docs/ADR/
tasks/TASK_TEMPLATE.md
tasks/SESSION_HANDOFF_TEMPLATE.md
Makefile
scripts/docs/
```

Intelligence Harness 先只補：

```text
docs/MENTAL_MODELS.md
docs/INVARIANTS.md
docs/FAILURE_MODES.md
```

## Level 3 — 高複雜度 / 高風險 / 研究型專案

適合交易系統、策略研究平台、多人協作 repo、正式產品。

可啟用完整架構：Control + Navigation + Doc Sync + Intelligence + Context Resilience。

## Harness Budget 評估

每個 repo 問 5 個問題：

1. 這個 repo 會活超過一個月嗎？
2. 會不會有 AI 多次修改？
3. 有沒有業務邏輯 / 金融邏輯 / 資料語義？
4. 有沒有前後端 / API / DB / pipeline 分層？
5. 壞掉時代價高不高？

```text
0~1 分：README + AGENTS
2~3 分：Minimal Viable Harness
4 分：加 Doc Sync Harness
5 分：加 Intelligence + Context Resilience Harness
```


---

<!-- FILE: 00_PLAYBOOK/02_IMPLEMENTATION_PHASES.md -->

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


---

<!-- FILE: 00_PLAYBOOK/03_CONTEXT_LIMITS_AND_RESILIENCE.md -->

# Context Limits and Resilience

## 核心原則

不要把 ChatGPT、Claude、Codex 的對話上下文當作專案記憶。

對話框只是工作台，repo 才是正式記憶。

## 三種記憶

```text
Context window = 這次回覆能直接看到的內容
Memory = 跨對話保留的高層次個人化資訊
Repo docs = 專案可靠 source of truth
```

只有 repo docs + git + tests 可以作為專案事實來源。

## 對話壓縮不能省略的資訊

壓縮時不得省略：

1. Current branch and commit
2. Active task
3. Do-not-touch files
4. Business logic changes
5. Open risks
6. Failed tests
7. Tests skipped with reason
8. Docs that need updates
9. ADRs required
10. Human decisions not yet made
11. Rollback plan
12. Uncertainty

永遠不要把「未知」壓縮成「確定」。

## Session 開始規則

每個 AI session 開始必須：

1. Read `AI_CONTEXT.md`
2. Read `docs/CURRENT_STATE.md`
3. Read `docs/CONTEXT_INDEX.md`
4. Read relevant `docs/CONTEXT_PACKS/*.md`
5. Read task spec
6. Run `git status --short`
7. Locate before edit

## Session 結束規則

每個 AI session 結束必須產生 Context Handoff。


---

<!-- FILE: 04_PROMPTS/harness_assessment_prompt.md -->

請評估目前 repo 的 AI Harness 等級，並只補「最小足夠」的 Context + Harness。

請不要一次建立所有可能文件。

原則：

1. 如果文件沒有明確讀者、更新時機、source of truth 角色，就不要新增。
2. 如果專案目前沒有該複雜度，就先列為 deferred。
3. 優先建立可以幫助人類定位與維護的文件，而不是長篇說明。
4. 優先建立：AI_CONTEXT.md、AGENTS.md、CLAUDE.md、ARCHITECTURE.md、FEATURE_MAP.md、RUNBOOK.md、Makefile。
5. 只有在 repo 已有明確 API / UI / data flow 時，才新增 API_MAP / UI_MAP / DATA_FLOW。
6. 只有在業務邏輯會頻繁變動時，才新增 DOMAIN_RULES / DOC_IMPACT_MATRIX。
7. 只有在策略研究或實驗很多時，才新增 HYPOTHESIS_LEDGER / EXPERIMENT_REGISTRY。
8. 只有在上下文或 handoff 已經成為問題時，才新增 CURRENT_STATE / CONTEXT_INDEX / CONTEXT_PACKS。

完成後請輸出：
- 選擇的 harness level
- 為什麼是這個 level
- 新增哪些文件
- 哪些文件刻意不新增
- deferred harness items
- 後續什麼情況下才要補 deferred items


---

<!-- FILE: 04_PROMPTS/bootstrap_minimal_harness.md -->

請為目前 repo 建立 Minimal Viable AI Harness。

本次只做文件 / harness / 開發流程，不要修改業務邏輯。

請新增或更新：

1. `AI_CONTEXT.md`
2. `AGENTS.md`
3. `CLAUDE.md`
4. `docs/ARCHITECTURE.md`
5. `docs/FEATURE_MAP.md`
6. `docs/RUNBOOK.md`
7. `Makefile`

限制：
- 不要重構 code。
- 不要新增不必要 dependency。
- 不要假裝不存在的測試已經存在。
- 如果某 target 目前無法完整實作，可以建立 placeholder 並清楚標註 TODO。

完成後請輸出：
- files added
- files changed
- chosen harness level
- deferred items
- tests/checks run
- rollback plan


---

<!-- FILE: 04_PROMPTS/add_doc_sync_harness.md -->

請在目前 repo 補上 Doc Sync Harness。

目標：當業務邏輯、API contract、資料流、UI 行為或 DB schema 發生改變時，AI 必須能判斷哪些文件需要同步更新，避免文件老化。

本次是治理 / 文件 / harness 任務，不要修改核心業務邏輯。

請完成：

1. 新增 `docs/DOMAIN_RULES.md`
2. 新增 `docs/DOC_IMPACT_MATRIX.md`
3. 新增 `docs/CHANGE_MANIFEST_TEMPLATE.md`
4. 新增 `docs/ADR/README.md`
5. 新增 `scripts/docs/check_doc_impact.py`
6. 更新 `Makefile`，新增 `docs-impact`
7. 更新 `AGENTS.md` / `CLAUDE.md`，加入 Doc Sync Required 規則
8. 更新 handoff templates，加入 Doc Sync Status

完成後執行：
make docs-check
make docs-impact
make verify

最後輸出：
Implementation summary
Files added
Files changed
Doc sync mechanism
Warnings
Rollback plan
Questions for human review


---

<!-- FILE: 04_PROMPTS/add_intelligence_harness.md -->

請在目前 repo 補上 Intelligence Harness。

目標：AI 不只是執行任務，而是讓人類累積理解、判斷力與設計能力。

本次不要修改業務邏輯。

請依 repo 複雜度選擇最小足夠版本，不要建立空泛文件。

優先新增：

1. `docs/MENTAL_MODELS.md`
2. `docs/INVARIANTS.md`
3. `docs/FAILURE_MODES.md`

如果此 repo 是研究 / 策略 / 實驗型，再新增：

4. `docs/HYPOTHESIS_LEDGER.md`
5. `docs/EXPERIMENT_REGISTRY.md`
6. `docs/GOLDEN_CASES.md`

如果此 repo 需要 AI 主動挑戰設計，再新增：

7. `docs/CRITIQUE_PROTOCOL.md`
8. `docs/QUESTION_BANK.md`

更新 AGENTS / CLAUDE / SESSION_HANDOFF_TEMPLATE。

完成後輸出：
- added files
- intentionally deferred files
- how this helps human learning
- how to maintain without doc bloat


---

<!-- FILE: 04_PROMPTS/add_context_resilience_harness.md -->

請在目前 repo 補上 Context Resilience Harness。

目標：避免長對話、上下文壓縮、換 session、換模型時遺漏重要資訊。

本次不要修改業務邏輯。

請新增：

1. `docs/CURRENT_STATE.md`
2. `docs/CONTEXT_INDEX.md`
3. `docs/CONTEXT_BUDGET.md`
4. `docs/COMPRESSION_RULES.md`
5. `docs/CONTEXT_PACKS/`
6. `tasks/CONTEXT_HANDOFF_TEMPLATE.md`

更新 AGENTS / CLAUDE / docs README / SESSION_HANDOFF_TEMPLATE。

要求：
- AI 不得把 chat history 當 source of truth。
- 每個 session 開始要讀 CURRENT_STATE / CONTEXT_INDEX。
- 每個 session 結束要提供 Context Handoff。
- 壓縮時不能省略風險、失敗測試、待決策、rollback plan。

完成後輸出：
- files added
- context loading strategy
- compression rules
- known limitations
- next recommended task
