---
created_at: 2026-06-12
updated_at: 2026-06-12
status: current
owner: ai-maintained
purpose: help the human learn the project through check questions
when_to_read: TBD
when_to_update: TBD
---

# QUESTION_BANK


## Architecture

1. What are the main layers of the system?
2. Which layer owns business logic?
3. Which layer should not know about UI?

## Frontend

1. Where does the UI call backend APIs?
2. Which file renders the main chart/form/page?
3. What should never be computed in the frontend?

## API

1. Which endpoints are read-only?
2. Which endpoints mutate state?
3. Which schema changes would break frontend?

## Data

1. What is raw data?
2. What is canonical data?
3. How are missing or duplicated records handled?

## Core Logic

1. What invariants must never break?
2. What tests prove the core behavior?
3. Where should you start debugging without AI?
