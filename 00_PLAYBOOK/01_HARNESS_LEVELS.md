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
