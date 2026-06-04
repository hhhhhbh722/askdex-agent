# -*- coding: utf-8 -*-
"""SMW + 详情页 wikitext 合并爬取，全字段不限制。"""
import json, os, re, shutil, time, requests, unicodedata

API = "https://wiki.biligame.com/rocom/api.php"
H = {"User-Agent": "Mozilla/5.0 (Win64; x64) AppleWebKit/537.36"}
ROOT = "D:/projects/wiki"

FIELDS_SMW = ["编号","名称","属性","生命","物攻","魔攻","物防","魔防","速度",
              "身高","体重","特性名称","特性描述","进化条件","蛋组","技能","技能石",
              "血脉技能","图鉴描述","精灵描述","精灵形态","分类"]

def safe(s):
    if not s: return "unknown"
    return "".join(c for c in str(s) if not unicodedata.category(c).startswith("C")).strip() or "unknown"

# Clear
if os.path.exists(ROOT):
    for item in os.listdir(ROOT):
        p = os.path.join(ROOT, item)
        if os.path.isdir(p): shutil.rmtree(p)

# Step 1: Fetch all from SMW with full fields
print("Step 1: SMW fetch...")
smw_data = {}
for offset in [0, 200, 400, 600]:
    q = "[[Category:精灵]]|?" + "|?".join(FIELDS_SMW) + f"|limit=200|offset={offset}"
    r = requests.get(f"{API}?action=ask&query={q}&format=json", headers=H, timeout=30)
    if r.status_code != 200:
        print(f"BLOCKED at offset {offset}")
        break
    batch = r.json().get("query", {}).get("results", {})
    smw_data.update(batch)
    print(f"  offset {offset}: +{len(batch)} total {len(smw_data)}")
    time.sleep(3)

with open("D:/projects/smw_full.json", "w", encoding="utf-8") as f:
    json.dump(smw_data, f, ensure_ascii=False)
print(f"SMW cached: {len(smw_data)} entries")

# Step 2: For each Pokemon, fetch wikitext for extra fields
print("Step 2: Detail pages...")
cats = {}
count = 0
for pn, pd in smw_data.items():
    pn_clean = pn.replace("​","").strip()
    if not pn_clean: continue
    po = dict(pd.get("printouts", {}))

    # ----- SMW data -----
    out = {}
    for k, vals in po.items():
        if not vals: continue
        for v in vals:
            if isinstance(v, dict): v = v.get("fulltext", str(v))
            v = str(v).replace("<desc_id=", "[").replace("</>", "]")
            if v and v != "None":
                out.setdefault(k, []).append(v)

    # ----- Wikitext data (extra fields) -----
    try:
        time.sleep(0.5)  # rate limit
        r2 = requests.get(f"{API}?action=parse&page={pn_clean}&prop=wikitext&format=json",
                          headers=H, timeout=15)
        if r2.status_code == 200:
            wt = r2.json().get("parse", {}).get("wikitext", {}).get("*", "")
            m = re.search(r'\{\{(?:宠物信息|精灵信息)[/\n\|][\s\S]*?\n\}\}', wt)
            if m:
                for match in re.finditer(r'\|\s*([^=|\n]+?)\s*=\s*(.*?)(?=\n\||\n\}\})', m.group(0), re.S):
                    k = safe(match.group(1))
                    v = re.sub(r'<[^>]+>', '', match.group(2)).strip()
                    if k and v and v != "None":
                        out.setdefault(k, []).append(v)
    except:
        pass

    # ----- Build output (deduplicate) -----
    lines = [f"# {pn_clean}"]
    seen_pairs = set()
    for k, vals in out.items():
        for v in vals:
            pair = (k, v)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            lines.append(f"{k}: {v}")

    # Folder by attribute
    attrs_raw = po.get("属性", []) or []
    attrs = []
    for a in attrs_raw:
        if isinstance(a, dict): a = a.get("fulltext", "")
        a2 = safe(a)
        if a2 and a2 not in attrs: attrs.append(a2)
    primary = attrs[0] if attrs else "unknown"

    folder = os.path.join(ROOT, primary)
    os.makedirs(folder, exist_ok=True)
    fname = safe(pn_clean)[:80]
    with open(os.path.join(folder, f"{fname}.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    cats[primary] = cats.get(primary, 0) + 1
    count += 1
    if count % 50 == 0:
        print(f"  {count}/{len(smw_data)}...")

print(f"\nDone! {count} files:")
for c, n in sorted(cats.items(), key=lambda x: -x[1]):
    print(f"  {c}/: {n} 只")
print(f"Total: {sum(cats.values())}")
