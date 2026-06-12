---
title: Legacy Safety And Compliance Note
owner: human
status: legacy
last_updated: 2026-06-12
---

# Safety And Compliance

Canonical safety rules are now in `docs/DOMAIN_RULES.md` and `docs/INVARIANTS.md`. This file is retained as the original short note.

Hackathon Hunter is a planning system. It must never represent that the user accepted rules, posted on social media, or submitted a final entry unless the user explicitly performs that action.

Required gates:

- Register for a hackathon.
- Select a project idea.
- Publish a repository or demo.
- Publish a social post.
- Submit the final entry.

Rules:

- Do not commit real API keys, tokens, cookies, or private keys.
- Do not treat LLM guesses as facts.
- Do not rank expired hackathons as active candidates.
- Do not enter build stage when AI policy is unknown.
- Disclose demo data and synthetic fallbacks in project materials.
