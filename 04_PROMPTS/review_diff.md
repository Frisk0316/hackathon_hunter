請 review 目前 uncommitted diff。

請檢查：

1. 是否超出原任務範圍
2. 是否有不必要重構
3. 是否新增 dependency
4. 是否破壞 API contract
5. 是否改變 business logic
6. 是否有 DB schema 風險
7. 是否缺少測試
8. 是否缺少文件更新
9. 是否需要 ADR
10. 是否 rollback 困難
11. 是否可以拆成更小 commit
12. 是否有上下文 / handoff 遺漏

最後給出：
- Safe to commit / Not safe to commit
- 必須修正項目
- 建議修正項目
- 可延後處理項目
- Doc Sync Status
- Human Learning Notes
