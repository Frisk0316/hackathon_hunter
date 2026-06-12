#!/usr/bin/env python3
"""
Lightweight doc impact checker.

Usage:
  python scripts/docs/check_doc_impact.py
  python scripts/docs/check_doc_impact.py --base main
"""
from __future__ import annotations
import argparse, subprocess

DEFAULT_RULES = [
    {"name":"frontend","prefixes":["frontend/","src/frontend/","web/"],"docs":["docs/UI_MAP.md","docs/FEATURE_MAP.md","docs/API_MAP.md","docs/CHANGELOG_AI.md"]},
    {"name":"api","prefixes":["src/api/","app/api/","backend/api/","routes","controllers"],"docs":["docs/API_MAP.md","docs/FEATURE_MAP.md","docs/DATA_FLOW.md","docs/CHANGELOG_AI.md"]},
    {"name":"core_logic","prefixes":["src/core/","src/domain/","app/core/","services/"],"docs":["docs/DOMAIN_RULES.md","docs/DOC_IMPACT_MATRIX.md","docs/CHANGELOG_AI.md","docs/ADR/"]},
    {"name":"data","prefixes":["src/data/","src/db/","db/","migrations/","ingest","pipeline"],"docs":["docs/DATA_FLOW.md","docs/DOMAIN_RULES.md","docs/RUNBOOK.md","docs/ADR/"]},
    {"name":"infra","prefixes":["Dockerfile","docker-compose",".github/","Makefile","scripts/"],"docs":["docs/RUNBOOK.md","AI_CONTEXT.md","AGENTS.md"]},
]

def git(args):
    try:
        out = subprocess.check_output(["git", *args], text=True, stderr=subprocess.STDOUT)
        return [x.strip() for x in out.splitlines() if x.strip()]
    except Exception as e:
        print(f"WARNING: git command failed: git {' '.join(args)} ({e})")
        return []

def changed_files(base):
    if base:
        return git(["diff","--name-only",f"{base}...HEAD"])
    return sorted(set(git(["diff","--name-only"]) + git(["diff","--cached","--name-only"])))

def matches(path, prefixes):
    return any(path.startswith(p) or p in path for p in prefixes)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--base', default=None)
    ap.add_argument('--strict', action='store_true')
    ns=ap.parse_args()
    changed=changed_files(ns.base)
    if not changed:
        print('No changed files detected.')
        return 0
    expected={}
    for f in changed:
        for r in DEFAULT_RULES:
            if matches(f, r['prefixes']):
                expected.setdefault(r['name'], set()).update(r['docs'])
    docs_changed={f for f in changed if f.endswith('.md') or f.startswith('docs/')}
    missing=[]
    print('Changed files:')
    for f in changed: print('  -', f)
    print('\nInferred areas:')
    if not expected:
        print('  - none')
    for area, docs in expected.items():
        print('  -', area)
        for d in sorted(docs):
            ok = d in changed or (d.endswith('/') and any(x.startswith(d) for x in docs_changed))
            print(f"      {'changed' if ok else 'missing'}: {d}")
            if not ok: missing.append((area,d))
    if missing:
        level='ERROR' if ns.strict else 'WARNING'
        print(f'\n{level}: possible missing documentation updates:')
        for area,d in missing: print(f'  - area={area} expected={d}')
        return 1 if ns.strict else 0
    print('\nPASS: documentation impact appears covered.')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
