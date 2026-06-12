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
