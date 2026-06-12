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
