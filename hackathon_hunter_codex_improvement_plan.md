# hackathon_hunter 改造規格：從一次性黑客松專案生成器升級為可循環執行的 Hackathon Intelligence Pipeline

> 給 Codex / Claude Code 的任務說明。請在 `Frisk0316/hackathon_hunter` repo 中實作。本文件的重點不是再生成一個黑客松作品，而是把現有 repo 改造成「可重複搜尋、篩選、分析、開案、產出投遞草稿、保留人工審核閘門」的自動化系統。

---

## 0. 當前 repo 狀態判斷

目前 repo 比較像是：

1. `CLAUDE.md`：一份 Claude Code 用的高階 prompt / 操作手冊。
2. `hackathons.json`：某一次搜尋出的黑客松候選清單。
3. `strategy.json`：某一次選題策略。
4. `hackathon-project/`、`hackathon-project-algofest/`：根據候選黑客松生成出的專案樣板或作品。

問題是，這些檔案目前多半是「一次性產物」，不是可穩定重跑的 pipeline。請把專案改造成有清楚 CLI、資料模型、資料來源、紀錄、測試、CI、人工閘門與成本/風險控管的系統。

---

## 1. 改造目標

把 repo 改造成以下工作流：

```text
schedule / manual trigger
  -> collect hackathons
  -> normalize + deduplicate
  -> eligibility/rules check
  -> score + rank
  -> analyze winners / judging context
  -> generate project ideas
  -> human approval gate
  -> create project spec + repo scaffold
  -> hand off coding tasks to Codex / Claude Code
  -> QA + submission package generation
  -> human final submission gate
  -> record results and update scoring weights
```

Codex / Claude Code 不應直接自動提交黑客松表單。系統可以產出 submission draft，但最後提交必須保留人工確認。

---

## 2. 新增建議工作流

### 2.1 Hackathon Radar Workflow

目的：定期搜尋近期、線上可參加、有獎金的黑客松。

新增 CLI：

```bash
python -m hackathon_hunter collect --days-ahead 90 --min-prize-usd 1000 --online-only
```

輸出：

```text
data/raw/{source}/{YYYYMMDD_HHMMSS}.json
data/processed/hackathons_{YYYYMMDD}.json
reports/radar_{YYYYMMDD}.md
```

資料來源先做 adapter interface，再逐步支援：

- Devpost
- DoraHacks
- Lablab.ai
- ETHGlobal
- MLH
- Unstop
- HackerEarth
- Gitcoin / Grants / Bounties 類平台
- 一般 web search fallback

每一筆 hackathon 都要保留：

```json
{
  "id": "stable_slug_or_hash",
  "name": "...",
  "platform": "devpost",
  "url": "...",
  "rules_url": "...",
  "deadline": "2026-07-01T23:59:00-04:00",
  "deadline_timezone": "America/New_York",
  "format": "online|hybrid|in_person|unknown",
  "prize_total_usd": 10000,
  "cash_prize": true,
  "tracks": [],
  "sponsors": [],
  "required_apis": [],
  "judging_criteria": [],
  "ai_policy": "allowed|restricted|forbidden|unknown",
  "eligibility": {
    "region_restricted": false,
    "student_only": false,
    "team_required": false
  },
  "source_evidence": [
    {
      "field": "deadline",
      "url": "...",
      "quote": "...",
      "fetched_at": "2026-06-04T12:00:00+08:00",
      "confidence": 0.95
    }
  ],
  "status": "open|upcoming|closed|unknown"
}
```

驗收條件：

- 不得把已截止黑客松列為 active candidate。
- deadline 必須含 timezone；若未知，標記 `confidence < 0.7`。
- 每個關鍵欄位都要有 `source_evidence`，不能只靠 LLM 推測。
- 若資料來源抓取失敗，保留 error log，不要直接吞掉。

---

### 2.2 Eligibility & Rules Checker Workflow

目的：防止投入不適合的黑客松。

新增 CLI：

```bash
python -m hackathon_hunter check-rules --input data/processed/hackathons_latest.json
```

檢查項目：

- 是否允許線上參與。
- 是否有地區限制。
- 是否 student-only / company-only。
- 是否明確允許 AI-assisted development。
- 是否需要現場 pitch。
- 是否需要指定平台帳號，例如 Zerve、DoraHacks wallet、Devpost profile。
- 是否需要 sponsor API / SDK。
- 是否有 IP / open-source / license 條款。
- 是否有社群貼文、影片、deck、demo URL 等提交要求。

輸出：

```text
reports/rules_check_{hackathon_id}.md
```

資料欄位：

```json
{
  "eligible": true,
  "blocking_issues": [],
  "warnings": [],
  "submission_requirements": {
    "github_repo": true,
    "demo_url": true,
    "video": true,
    "deck": false,
    "social_post": false,
    "public_profile": true
  },
  "human_review_required": true
}
```

驗收條件：

- AI policy 為 `unknown` 時，不得自動進入 build 階段。
- 有 region / student / on-site 限制時，必須標示為 blocking 或 warning。
- 每個 blocking issue 都要附來源 URL 或明確說明「來源不足」。

---

### 2.3 Scoring & Ranking Workflow

目的：把黑客松候選轉成可比較的排序，而不是憑直覺選。

新增 CLI：

```bash
python -m hackathon_hunter rank --input data/processed/hackathons_latest.json --profile config/user_profile.yaml
```

請不要再使用過度確定的「勝率 = 22%」這種估計。改用多維分數：

```json
{
  "roi_score": 0.78,
  "feasibility_score": 0.82,
  "strategic_fit_score": 0.88,
  "evidence_quality_score": 0.73,
  "delivery_risk_score": 0.31,
  "overall_score": 0.80,
  "ranking_reason": "..."
}
```

建議 scoring 權重：

```yaml
weights:
  prize_cash: 0.15
  online_allowed: 0.15
  deadline_buffer: 0.15
  ai_policy_clear: 0.15
  sponsor_api_fit: 0.15
  past_winner_analyzable: 0.10
  low_submission_estimate: 0.05
  user_domain_fit: 0.10
```

驗收條件：

- 排名報告要同時列出 top picks 與 rejected candidates。
- 所有分數必須可追溯到欄位與權重。
- 若資料不足，降低 `evidence_quality_score`，不要假裝知道。

---

### 2.4 Winner Intelligence Workflow

目的：分析過去得獎作品與評審偏好，避免只做技術上可行但評審不買單的作品。

新增 CLI：

```bash
python -m hackathon_hunter analyze-winners --hackathon-id <id>
```

輸出：

```text
reports/winners/{hackathon_id}.md
```

分析內容：

- 過去 winners / finalists。
- 作品型態：web app、API、agent、dashboard、mobile app、research notebook。
- demo 型態：影片、live demo、GitHub README、deck。
- 評審 rubric 對應：innovation / impact / technical / design / sponsor API usage。
- 得獎作品共通特徵。
- 本次應該避開的專案型態。
- 推薦 project archetypes。

驗收條件：

- 至少輸出 3 個得獎脈絡假說。
- 每個假說要有 evidence 或標示 confidence。
- 不要只摘要 winners，要轉成「本次專案策略」。

---

### 2.5 Idea Generation & Selection Workflow

目的：針對每個高分黑客松產生多個可行專案方向。

新增 CLI：

```bash
python -m hackathon_hunter ideate --hackathon-id <id> --n 5
```

輸出：

```text
reports/ideas/{hackathon_id}.md
strategy/{hackathon_id}_ideas.json
```

每個 idea 應包含：

```json
{
  "name": "...",
  "tagline": "...",
  "problem": "...",
  "target_user": "...",
  "why_now": "...",
  "sponsor_api_usage": "...",
  "mvp_scope": ["..."],
  "demo_flow": "...",
  "tech_stack": ["..."],
  "risks": [],
  "estimated_build_hours": 24,
  "feasibility_score": 0.8,
  "judging_fit_score": 0.85,
  "differentiation_score": 0.7
}
```

驗收條件：

- 不得只輸出一個 idea。
- 每個 idea 要有 MVP scope 與不做什麼。
- 必須產出 `human_decision_required: true`，由使用者挑選後才建 repo。

---

### 2.6 Project Spec Builder Workflow

目的：把選定的 idea 轉成 Codex / Claude Code 可執行的 engineering spec。

新增 CLI：

```bash
python -m hackathon_hunter build-spec --hackathon-id <id> --idea-id <idea_id>
```

輸出：

```text
projects/{hackathon_id}/{project_slug}/SPEC.md
projects/{hackathon_id}/{project_slug}/TASKS.md
projects/{hackathon_id}/{project_slug}/README_DRAFT.md
projects/{hackathon_id}/{project_slug}/SUBMISSION_DRAFT.md
```

`TASKS.md` 應拆成 Codex/Claude 可逐步完成的任務：

```markdown
# TASKS

## Milestone 1 — Skeleton
- [ ] create app scaffold
- [ ] add env example
- [ ] add basic CI

## Milestone 2 — Core demo path
- [ ] implement primary user flow
- [ ] add sponsor API integration
- [ ] add fallback error handling

## Milestone 3 — Submission quality
- [ ] README complete
- [ ] demo script complete
- [ ] screenshots generated
- [ ] smoke test passes
```

驗收條件：

- Spec 必須可獨立交給 coding agent。
- 不得把真實 API key 寫進檔案。
- 必須列出 human approval points。

---

### 2.7 Coding Agent Handoff Workflow

目的：讓 Codex / Claude Code 專心做「coding worker」，而不是負責全部決策。

新增指令或文件：

```text
projects/{hackathon_id}/{project_slug}/AGENT_BRIEF.md
```

內容包含：

- Project goal。
- Non-goals。
- Required APIs。
- Demo path。
- Acceptance tests。
- Submission requirements。
- Time budget。
- Files not to touch。

驗收條件：

- `AGENT_BRIEF.md` 一定要短於 200 行。
- 要有明確 done criteria。
- coding agent 完成後要產生 `HANDOFF_REPORT.md`。

---

### 2.8 QA / Submission Package Workflow

目的：在提交前檢查作品是否真的能評審。

新增 CLI：

```bash
python -m hackathon_hunter qa --project projects/<hackathon_id>/<project_slug>
```

檢查項目：

- `README.md` 是否有 Demo URL、Quick Start、Tech Stack、Architecture。
- `.env.example` 是否存在。
- 是否有測試或至少 smoke test。
- demo URL 是否可訪問。
- API endpoint 是否可回應。
- README 中是否仍有 `TODO`、`VIDEO_URL`、`YOUR_REPO`。
- 是否誤提交 `.env`、token、private key。
- 是否明確標示 synthetic data / demo data。

輸出：

```text
projects/<hackathon_id>/<project_slug>/QA_REPORT.md
projects/<hackathon_id>/<project_slug>/SUBMISSION_PACKAGE.md
```

驗收條件：

- 若 README 有 placeholder，QA 必須 fail。
- 若 synthetic fallback 被使用，submission draft 必須揭露。
- 不得自動 submit，只能產出投遞草稿。

---

### 2.9 Results Feedback Workflow

目的：讓每一輪結果回饋到選題策略，而不是無腦重複。

新增 CLI：

```bash
python -m hackathon_hunter record-result --hackathon-id <id> --outcome submitted|finalist|winner|rejected|abandoned
```

輸出：

```text
logs/rounds/{YYYYMMDD}_{hackathon_id}.json
reports/retrospectives/{hackathon_id}.md
```

欄位：

```json
{
  "hackathon_id": "...",
  "project_slug": "...",
  "hours_spent": 38,
  "api_cost_usd": 24.3,
  "infra_cost_usd": 0.0,
  "submitted": true,
  "outcome": "finalist",
  "what_worked": [],
  "what_failed": [],
  "scoring_adjustments": {}
}
```

驗收條件：

- 每個完成/放棄的專案都要有 retrospective。
- 下次 ranking 可以讀取歷史結果調整權重。

---

## 3. 專案結構重整

請改成下列結構：

```text
hackathon_hunter/
├── README.md
├── CLAUDE.md
├── pyproject.toml
├── .env.example
├── .gitignore
├── Makefile
├── config/
│   ├── sources.yaml
│   ├── scoring.yaml
│   └── user_profile.example.yaml
├── data/
│   ├── raw/.gitkeep
│   └── processed/.gitkeep
├── docs/
│   ├── architecture.md
│   ├── workflow.md
│   └── safety_and_compliance.md
├── hackathon_hunter/
│   ├── __init__.py
│   ├── cli.py
│   ├── models.py
│   ├── storage.py
│   ├── scoring.py
│   ├── rules.py
│   ├── reports.py
│   ├── sources/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── devpost.py
│   │   ├── dorahacks.py
│   │   ├── lablab.py
│   │   └── web_search.py
│   └── workflows/
│       ├── collect.py
│       ├── rank.py
│       ├── analyze_winners.py
│       ├── ideate.py
│       ├── build_spec.py
│       └── qa.py
├── projects/
│   └── .gitkeep
├── reports/
│   └── .gitkeep
├── strategy/
│   └── .gitkeep
├── logs/
│   └── .gitkeep
└── tests/
    ├── test_models.py
    ├── test_scoring.py
    ├── test_rules.py
    └── fixtures/
```

現有 `hackathon-project/` 和 `hackathon-project-algofest/` 請不要刪除，但請移到：

```text
archive/generated-projects/
```

或在 README 中標記為 historical examples，避免和 hunter 本體混在一起。

---

## 4. 必要技術實作

### 4.1 Python package

新增 `pyproject.toml`，建議使用：

```toml
[project]
name = "hackathon-hunter"
version = "0.1.0"
description = "Hackathon discovery, ranking, and project-spec automation pipeline"
requires-python = ">=3.10"
dependencies = [
  "pydantic>=2",
  "typer>=0.12",
  "httpx>=0.27",
  "beautifulsoup4>=4.12",
  "python-dateutil>=2.9",
  "pyyaml>=6",
  "rich>=13",
]

[project.optional-dependencies]
dev = ["pytest", "ruff", "mypy"]

[project.scripts]
hackathon-hunter = "hackathon_hunter.cli:app"
```

### 4.2 Pydantic models

在 `models.py` 定義：

- `Hackathon`
- `Evidence`
- `Eligibility`
- `Prize`
- `SubmissionRequirements`
- `ScoreBreakdown`
- `ProjectIdea`
- `RunLog`

所有 deadline 要用 timezone-aware datetime。

### 4.3 Storage

先用 JSONL / SQLite 都可。MVP 建議：

```text
data/raw/*.json
 data/processed/*.json
logs/*.jsonl
```

必要功能：

- `save_raw_snapshot(source, payload)`
- `load_latest_processed()`
- `save_report(path, markdown)`
- `deduplicate_hackathons(items)`

### 4.4 CI

新增 GitHub Actions：

```text
.github/workflows/ci.yml
.github/workflows/nightly_radar.yml
```

`ci.yml`：

- install deps
- run ruff
- run pytest
- validate JSON schema

`nightly_radar.yml`：

- 可手動觸發 `workflow_dispatch`
- 可排程每日或每週執行
- 只產生報告，不自動建立黑客松專案
- 不自動提交外部平台表單

---

## 5. 修正現有內容的具體問題

### 5.1 `hackathons.json` 與 `strategy.json` 過期問題

目前這兩個檔案應該改為：

```text
examples/hackathons.sample.json
examples/strategy.sample.json
```

並在 README 註明它們只是 sample，不是 active candidates。

新增 freshness check：

- 若 deadline < now，標示 `status = closed`。
- active ranking 必須排除 closed。
- report 中列出 expired candidates 供回顧，不可列為推薦。

### 5.2 不要混淆 hunter 與 generated project

現在 repo 同時含 hunter 指令與 PredictPulse 專案，容易讓 agent 以為主要任務是繼續改 PredictPulse。請把生成專案移到 archive，並建立 root README 說明：

```text
This repository is the automation system, not a single hackathon submission.
Generated projects live under archive/generated-projects or separate repos.
```

### 5.3 假資料與真實資料要明確分流

如果某個 generated project 使用 synthetic fallback，必須：

- 在 README 標示 demo / synthetic data mode。
- 在 submission draft 標示資料來源。
- QA 若發現使用 synthetic data 但未揭露，直接 fail。

### 5.4 README placeholder 檢查

QA 必須搜尋：

```text
TODO
TBD
VIDEO_URL
DEMO_URL
YOUR_REPO
PLACEHOLDER
```

發現就 fail。

### 5.5 重複章節與文件品質

若 README 有重複區塊，例如重複 `Key Findings`，QA 應 warning 或 fail。

---

## 6. 人工閘門設計

系統必須保留以下人工確認點：

1. 是否報名該黑客松。
2. 是否採用某個 project idea。
3. 是否公開 repo / demo。
4. 是否產生社群貼文。
5. 是否正式提交。

可自動完成：

- 搜尋。
- 初步篩選。
- 規則摘要。
- 得獎作品分析。
- project idea generation。
- spec / task / README draft。
- QA 檢查。
- submission draft。

不可自動完成：

- 代表使用者同意官方規則。
- 代表使用者提交法律/IP/原創性聲明。
- 代表使用者張貼社群貼文。
- 代表使用者投遞最終作品。

---

## 7. Root README 應新增內容

請建立新的 root `README.md`，內容至少包含：

```markdown
# Hackathon Hunter

Hackathon Hunter is a research and planning pipeline for discovering online prize hackathons, checking eligibility, ranking opportunities, analyzing winners, generating project specs, and preparing submission packages.

It does not automatically submit hackathon entries. Final submission always requires human approval.

## Workflows

1. Collect hackathons
2. Check rules and eligibility
3. Rank candidates
4. Analyze winners
5. Generate ideas
6. Build project specs
7. Hand off to coding agents
8. QA submission package
9. Record results

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
hackathon-hunter collect --days-ahead 90 --online-only --min-prize-usd 1000
hackathon-hunter rank
```

## Safety

- No auto-submit.
- No secret commits.
- Evidence required for deadlines, rules, prize, AI policy.
- Synthetic data must be disclosed.
```

---

## 8. 實作順序

請依照以下 PR / commit 順序實作。

### Phase 1 — Repo cleanup and package skeleton

- [ ] 新增 root README。
- [ ] 新增 `pyproject.toml`。
- [ ] 建立 `hackathon_hunter/` package。
- [ ] 建立 `models.py`、`cli.py`。
- [ ] 將現有 generated projects 移到 archive 或標記為 examples。
- [ ] 將 `hackathons.json` / `strategy.json` 移到 examples。

### Phase 2 — Data model and collect workflow

- [ ] 建立 source adapter interface。
- [ ] 實作至少一個 adapter：Devpost 或 generic web search/mock adapter。
- [ ] 建立 raw snapshot 與 processed output。
- [ ] 加入 deadline freshness check。
- [ ] 加入 dedup。

### Phase 3 — Rules and ranking

- [ ] 實作 rules checker。
- [ ] 實作 scoring model。
- [ ] 產出 ranking markdown report。
- [ ] 加入 evidence confidence。

### Phase 4 — Winner analysis and idea generation

- [ ] 產生 winner intelligence report template。
- [ ] 產生 project ideas JSON / MD。
- [ ] 加入 human approval gate 欄位。

### Phase 5 — Spec builder and coding-agent handoff

- [ ] 產生 `SPEC.md`。
- [ ] 產生 `TASKS.md`。
- [ ] 產生 `AGENT_BRIEF.md`。
- [ ] 產生 `SUBMISSION_DRAFT.md`。

### Phase 6 — QA and CI

- [ ] 實作 README placeholder check。
- [ ] 實作 secret scan basic check。
- [ ] 實作 demo URL / API smoke test interface。
- [ ] 新增 pytest。
- [ ] 新增 GitHub Actions CI。

---

## 9. 驗收標準

完成後，以下指令應能跑：

```bash
pip install -e .[dev]
pytest
ruff check .
hackathon-hunter collect --mock
hackathon-hunter rank --input data/processed/mock_hackathons.json
hackathon-hunter check-rules --input data/processed/mock_hackathons.json
hackathon-hunter ideate --hackathon-id mock-hackathon-001
hackathon-hunter build-spec --hackathon-id mock-hackathon-001 --idea-id idea-001
hackathon-hunter qa --project projects/mock-hackathon-001/idea-001
```

且應產生：

```text
data/raw/
data/processed/
reports/radar_*.md
reports/rules_check_*.md
reports/ideas/*.md
projects/*/*/SPEC.md
projects/*/*/TASKS.md
projects/*/*/AGENT_BRIEF.md
projects/*/*/SUBMISSION_DRAFT.md
projects/*/*/QA_REPORT.md
```

---

## 10. 重要限制

- 不要自動登入 Devpost / DoraHacks / Lablab 或代替使用者提交。
- 不要把 API key、token、session cookie 寫入 repo。
- 不要使用假資料冒充真實資料。
- 不要把 LLM 推測當成事實；所有 deadline、prize、rules、AI policy 必須附 evidence。
- 不要把已截止黑客松推薦為 active candidate。
- 不要只修 PredictPulse；本任務是改造 hunter 系統本身。

---

## 11. 建議先做的最小可行版本

最小 MVP 範圍：

1. CLI + Pydantic models。
2. mock source + one real source adapter。
3. freshness / deadline check。
4. rank report。
5. project spec builder。
6. QA placeholder checker。
7. GitHub Actions CI。

不需要在第一版完成：

- 自動填寫投稿表單。
- 全自動部署。
- 影片自動生成。
- 成本最佳化。
- 多 agent 平行開發。

第一版先讓系統可重複、可驗證、可被 Codex/Claude 接手，再逐步加自動化。
