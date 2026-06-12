---
created_at: 2026-06-12
updated_at: 2026-06-12
status: current
owner: ai-maintained
purpose: explain system structure and dependency direction
when_to_read: TBD
when_to_update: TBD
---

# ARCHITECTURE


## Overview

[用 5~10 行說明整體系統架構]

## Directory Map

```text
project-root/
  src/
  frontend/
  tests/
  scripts/
  docs/
```

## Layers

### Frontend / UI

Responsibility:
- [負責什麼]

Main files/directories:
- `[path]`

Must not:
- [不該做什麼]

### API / Interface Layer

Responsibility:
- [負責什麼]

### Service / Application Layer

Responsibility:
- [負責什麼]

### Core Domain Logic

Responsibility:
- [負責什麼]

### Data / Persistence Layer

Responsibility:
- [負責什麼]

## Dependency Direction

Allowed:

```text
UI -> API client -> API route -> service -> core logic -> data layer
```

Avoid:

```text
UI -> database
API route -> UI logic
core logic -> framework-specific UI code
data layer -> presentation logic
```

## Known Gaps

- [Gap]
