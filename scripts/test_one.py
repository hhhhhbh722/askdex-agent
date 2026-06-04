# -*- coding: utf-8 -*-
"""测试单只精灵：SMW + 详情页合并。"""
import json, requests, re, unicodedata, os, sys

API = "https://wiki.biligame.com/rocom/api.php"
H = {"User-Agent": "Mozilla/5.0 (Win64; x64) AppleWebKit/537.36"}
TARGET = sys.argv[1] if len(sys.argv) > 1 else "咔咔鸟"

def safe(s):
    return "".join(c for c in str(s) if not unicodedata.category(c).startswith("C")).strip() or "unknown"

# Step 1: SMW
q = f"[[{TARGET}]]|?属性|?生命|?物攻|?魔攻|?物防|?魔防|?速度|?特性描述|?技能|limit=1"
r = requests.get(f"{API}?action=ask&query={q}&format=json", headers=H, timeout=15)
smw_data = r.json().get("query", {}).get("results", {}) if r.status_code == 200 else {}

# Step 2: Wikitext
r2 = requests.get(f"{API}?action=parse&page={TARGET}&prop=wikitext&format=json", headers=H, timeout=15)
wt = r2.json().get("parse", {}).get("wikitext", {}).get("*", "") if r2.status_code == 200 else ""

# Merge
out = {}

# SMW
for pn, pd in smw_data.items():
    pn = pn.strip().replace("​", "")
    po = pd.get("printouts", {})
    for k, vals in po.items():
        if not vals: continue
        for v in vals:
            if isinstance(v, dict): v = v.get("fulltext", str(v))
            v = str(v).replace("<desc_id=", "[").replace("</>", "]")
            if v and v != "None":
                out.setdefault(k, []).append(v)

# Wikitext
if wt:
    # 多种模板名: {{宠物信息/精灵...}}, {{精灵信息...}}, {{宠物信息...}}
    m = re.search(r'\{\{(?:宠物信息|精灵信息)[/\n\|][\s\S]*?\n\}\}', wt)
    if m:
        block = m.group(0)
        for match in re.finditer(r'\|\s*([^=|\n]+?)\s*=\s*(.*?)(?=\n\||\n\}\})', block, re.S):
            k = safe(match.group(1))
            v = re.sub(r"<[^>]+>", "", match.group(2)).strip()
            if k and v and v != "None":
                out.setdefault(k, []).append(v)
        print(f"Wikitext added {sum(1 for v in out.values() if len(v)>0)} fields")
    else:
        print("Wikitext: no template found (page may use different format)")
else:
    print(f"Wikitext: HTTP {r2.status_code}")

# Output
attrs = []
attr_raw = out.get("属性", [[]])[0] if out.get("属性") else ""
for a in (attr_raw or "").replace("属性:", "").split("/"):
    a = safe(a.strip())
    if a: attrs.append(a)
if not attrs:
    for k in ["属性1", "属性2"]:
        if k in out:
            attrs.append(out[k][0])

primary = attrs[0] if attrs else "unknown"
pn_display = TARGET

lines = [f"# {pn_display}"]
for k, vals in out.items():
    for v in vals:
        lines.append(f"{k}: {v}")

folder = f"D:/projects/wiki/{primary}"
os.makedirs(folder, exist_ok=True)
fp = f"{folder}/{safe(pn_display)}.txt"
with open(fp, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"\nSaved to {fp}")
print(f"Total fields: {len(out)}")
print("---")
print("\n".join(lines))
