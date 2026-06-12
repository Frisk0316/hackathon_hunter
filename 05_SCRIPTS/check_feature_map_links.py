#!/usr/bin/env python3
"""Check repo-relative paths referenced in docs/FEATURE_MAP.md."""
from pathlib import Path
import re

def looks_like_path(s):
    if s.startswith(('http://','https://','mailto:')): return False
    if ' ' in s: return False
    return '/' in s or '.' in Path(s).name

def main():
    fm=Path('docs/FEATURE_MAP.md')
    if not fm.exists():
        print('WARNING: docs/FEATURE_MAP.md not found.')
        return 0
    text=fm.read_text(encoding='utf-8', errors='replace')
    refs=sorted({m.group(1).strip() for m in re.finditer(r'`([^`]+)`', text) if looks_like_path(m.group(1).strip())})
    missing=[r for r in refs if '[' not in r and ']' not in r and not Path(r).exists()]
    print(f'Checked referenced paths: {len(refs)}')
    if missing:
        print('WARNING: referenced paths not found:')
        for r in missing: print('  -', r)
    else:
        print('PASS: all referenced paths exist.')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
