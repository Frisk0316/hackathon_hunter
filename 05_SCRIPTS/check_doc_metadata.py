#!/usr/bin/env python3
"""Lightweight markdown metadata checker."""
from __future__ import annotations
import argparse
from pathlib import Path

def has_fm(text):
    return text.startswith('---\n') and '\n---\n' in text[4:]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--strict', action='store_true')
    ap.add_argument('--paths', nargs='*', default=['docs','iteration-docs'])
    ns=ap.parse_args()
    files=[]
    for root in ns.paths:
        p=Path(root)
        if p.exists(): files.extend(sorted(p.rglob('*.md')))
    if not files:
        print('No markdown files found.')
        return 0
    missing=[]
    for f in files:
        if not has_fm(f.read_text(encoding='utf-8', errors='replace')):
            missing.append(str(f))
    print(f'Checked markdown files: {len(files)}')
    if missing:
        level='ERROR' if ns.strict else 'WARNING'
        print(f'{level}: files missing YAML front matter:')
        for f in missing: print('  -', f)
        return 1 if ns.strict else 0
    print('PASS: all checked files have YAML front matter.')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
