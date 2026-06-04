# -*- coding: utf-8 -*-
"""Fetch 2 sample Pokemon with ALL SMW fields and display."""
import json, requests, unicodedata, os

API = 'https://wiki.biligame.com/rocom/api.php'
H = {'User-Agent': 'Mozilla/5.0 (Win64; x64) AppleWebKit/537.36'}

def safe(s):
    return ''.join(c for c in str(s) if not unicodedata.category(c).startswith('C') and c not in '<>:"/\\|?*').strip() or 'unknown'

# 加属性字段触发 dict 格式
r = requests.get(f'{API}?action=ask&query=[[Category:精灵]]|?属性|limit=2&format=json', headers=H, timeout=30)
if r.status_code != 200:
    print(f'HTTP {r.status_code} - WAF blocked')
    exit()

results = r.json().get('query', {}).get('results', {})

for pn, pd in list(results.items())[:2]:
    pn = safe(pn)
    po = pd.get('printouts', {})
    if not isinstance(po, dict):
        print(f'SKIP {pn}: printouts is {type(po)}')
        continue

    # Get attrs
    attr_raw = po.get('属性', []) or []
    attrs = []
    for a in attr_raw:
        if isinstance(a, dict):
            a = a.get('fulltext', '')
        if a:
            attrs.append(safe(a))
    primary = attrs[0] if attrs else 'unknown'

    # Dump ALL fields
    lines = [f'# {pn}']
    for key, vals in po.items():
        if not vals:
            continue
        for v in vals:
            if isinstance(v, dict):
                v = v.get('fulltext', str(v))
            v = str(v).replace('<desc_id=', '[').replace('</>', ']')
            if v and v != 'None':
                lines.append(f'{key}: {v}')

    fp = f'D:/projects/wiki/{primary}_{pn}.txt'
    os.makedirs('D:/projects/wiki', exist_ok=True)
    with open(fp, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f'=== {pn} ({primary}) ===')
    print('\n'.join(lines))
    print()
    print(f'Saved to {fp}')
