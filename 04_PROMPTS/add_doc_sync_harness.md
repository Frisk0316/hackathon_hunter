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
