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
