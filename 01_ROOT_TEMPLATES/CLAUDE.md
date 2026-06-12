@AGENTS.md

# Claude Code Specific Instructions

## Role

Claude Code is usually the implementation and reasoning agent for this repo, but must follow `AGENTS.md` and any existing collaboration/review flow.

## Planning

Use plan mode or explicit design analysis before editing for:

- architecture changes
- database changes
- multi-file changes
- deployment changes
- business logic changes
- refactoring
- unclear bug reports

## Do Not

- Do not bypass review flow.
- Do not commit unless explicitly asked.
- Do not claim tests passed unless actually run.
- Do not rely on chat history as source of truth.
- Do not silently change business rules.

## Handoff

Every session must end with a handoff summary and, when applicable, a Context Handoff.
