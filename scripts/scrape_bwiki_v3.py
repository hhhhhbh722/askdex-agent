# -*- coding: utf-8 -*-
"""爬取 BiliGame 洛克王国 Wiki → 完整精灵数据（HP/属性克制/技能/进化）+ 属性分类。"""
import json, os, re, time, shutil, requests

API = "https://wiki.biligame.com/rocom/api.php"
HEADERS = {"User-Agent": "Mozilla/5.0 (Win64; x64) AppleWebKit/537.36"}
ROOT = "D:/projects/wiki"

# 属性克制矩阵（洛克王国:世界）
TYPE_CHART = {
    "火": {"克": ["草","虫","冰","机械"], "抗": ["火","水","岩"], "被克": ["水","岩","地"]},
    "水": {"克": ["火","岩","地"], "抗": ["水","草","电"], "被克": ["草","电"]},
    "草": {"克": ["水","岩","地"], "抗": ["草","电","地"], "被克": ["火","虫","冰","飞行"]},
    "光": {"克": ["恶","幽"], "抗": ["光","幻"], "被克": ["草","幽"]},
    "恶": {"克": ["光","幻","超能"], "抗": ["恶","幽"], "被克": ["光","格斗"]},
    "普通": {"克": [], "抗": [], "被克": ["格斗"]},
    "电": {"克": ["水","飞行"], "抗": ["电"], "被克": ["地"]},
    "冰": {"克": ["草","地","飞行","龙"], "抗": ["冰"], "被克": ["火","格斗","岩","钢"]},
    "格斗": {"克": ["普通","冰","岩","钢","恶"], "抗": ["虫","岩"], "被克": ["飞行","超能","虫"]},
    "毒": {"克": ["草","虫"], "抗": ["毒","草","格斗"], "被克": ["地","超能"]},
    "地": {"克": ["火","电","毒","岩","钢"], "抗": ["岩","毒"], "被克": ["水","草","冰"]},
    "飞行":{"克": ["草","格斗","虫"], "抗": ["地","格斗","草"], "被克": ["电","冰","岩"]},
    "超能":{"克": ["格斗","毒"], "抗": ["超能"], "被克": ["虫","幽","恶"]},
    "虫": {"克": ["草","超能","恶"], "抗": ["草","地","格斗"], "被克": ["火","飞行","岩","毒"]},
    "岩": {"克": ["火","冰","飞行","虫"], "抗": ["普通","火","飞行","毒"], "被克": ["水","草","格斗","地","钢"]},
    "幽": {"克": ["超能","光"], "抗": ["普通","幽","毒"], "被克": ["恶","光"]},
    "龙": {"克": ["龙"], "抗": ["火","水","草","电"], "被克": ["冰","龙"]},
    "钢": {"克": ["冰","岩"], "抗": ["普通","草","虫","岩","冰","飞行","超能","龙","钢"], "被克":["火","格斗","地"]},
    "机械":{"克":["冰","岩","钢"], "抗":["普通","草","电","飞行","超能","虫","岩","冰","龙","钢","机械"],"被克":["火","水","格斗","地"]},
    "萌": {"克":["恶","幽","格斗"],"抗":["普通","萌"],"被克":["恶","幽"]},
    "幻": {"克":["光"],"抗":["光","幻","恶"],"被克":["幽"]},
    "首领":{"克":["普通","火","水","草","电","冰","格斗","毒","地","飞行","超能","虫","岩","幽","龙","钢","恶","光","幻","萌","机械"],"抗":["普通"],"被克":["普通","火","水","草","电","冰","格斗","毒","地","飞行","超能","虫","岩","幽","龙","钢","恶","光","幻","萌","机械"]},
}

# 清空
if os.path.exists(ROOT):
    for item in os.listdir(ROOT):
        p = os.path.join(ROOT, item)
        if os.path.isdir(p): shutil.rmtree(p)
        elif item.endswith(".txt"): os.remove(item)


def fetch_all_pages():
    """分段查询避免 WAF 拦截。"""
    all_results = {}
    offset = 0
    while True:
        q = "[[Category:精灵]]|?" + "|?".join([
            "编号","属性","HP","生命","物攻","魔攻","物防","魔防","速度",
            "身高","体重","进化条件","特性名称","特性描述","蛋组","分类",
        ]) + f"|limit=200|offset={offset}"
        r = requests.get(f"{API}?action=ask&query={q}&format=json", headers=HEADERS, timeout=30)
        if r.status_code != 200:
            print(f"  API error {r.status_code} at offset {offset}")
            break
        data = r.json()
        batch = data.get("query", {}).get("results", {})
        if not batch: break
        all_results.update(batch)
        offset += 200
        print(f"  Fetched {offset}...")
        time.sleep(2)
    return all_results


def fetch_wikitext(page_name: str) -> str:
    try:
        r = requests.get(API, params={"action":"parse","page":page_name,"prop":"wikitext","format":"json"},
                         headers=HEADERS, timeout=15)
        return r.json().get("parse",{}).get("wikitext",{}).get("*","")
    except: return ""


def parse_template(wt: str) -> dict:
    info = {}
    m = re.search(r'\{\{宠物信息/精灵[\s\S]*?\n\}\}', wt)
    if not m:
        m = re.search(r'\{\{宠物信息\s*\|[\s\S]*?\n\}\}', wt)
    if not m: return info
    block = m.group(0)
    for match in re.finditer(r'\|\s*([^=|\n]+?)\s*=\s*(.*?)(?=\n\||\n\}\})', block, re.S):
        k = match.group(1).strip()
        v = re.sub(r'<[^>]+>', '', match.group(2)).strip()
        v = re.sub(r'\{\{[^}]+\}\}', '', v)
        info[k] = v
    return info


def compute_effectiveness(attrs: list[str]) -> str:
    """根据精灵属性计算克制/被克关系。"""
    weak_to = set(); strong_vs = set(); resist = set()
    for a in attrs:
        a = a.strip()
        if a in TYPE_CHART:
            strong_vs.update(TYPE_CHART[a].get("克",[]))
            resist.update(TYPE_CHART[a].get("抗",[]))
            weak_to.update(TYPE_CHART[a].get("被克",[]))
    lines = []
    if strong_vs: lines.append(f"克制: {', '.join(sorted(strong_vs))}")
    if resist: lines.append(f"抵抗: {', '.join(sorted(resist))}")
    if weak_to: lines.append(f"被克制: {', '.join(sorted(weak_to))}")
    return "\n".join(lines)


def save_pokemon(name, data, smw):
    attrs = set()
    for k in ["属性","属性1","属性2"]:
        for a in re.split(r'[/,，/]', data.get(k,"") or ""):
            if a.strip(): attrs.add(a.strip())
    smw_attrs = smw.get("属性",[]) or []
    for a in smw_attrs:
        if a and a != "未知": attrs.add(a.strip())
    primary = list(attrs)[0] if attrs else "未知"
    folder = os.path.join(ROOT, primary)
    os.makedirs(folder, exist_ok=True)

    pid_raw = smw.get("编号",[name])
    pid = data.get("编号", data.get("编号名称","")) or (pid_raw[0] if pid_raw else name)

    lines = [f"# {name}", f"编号: {pid}", f"属性: {', '.join(attrs) if attrs else '未知'}"]

    # 种族值
    FIELD_MAP = {
        "生命":"HP","HP":"HP","物攻":"物攻","魔攻":"魔攻","物防":"物防","魔防":"魔防","速度":"速度",
    }
    stats = {}
    for wk, label in FIELD_MAP.items():
        v = data.get(wk,"") or ""
        if v:
            try: stats[label] = int(v)
            except: stats[label] = v
    if stats:
        for label in ["HP","物攻","魔攻","物防","魔防","速度"]:
            if label in stats:
                lines.append(f"{label}: {stats[label]}")
        total = sum(int(v) for v in stats.values() if isinstance(v,int))
        lines.append(f"种族值合计: {total}")

    for wk, label in [
        ("身高","身高"),("体重","体重"),("特性名称","特性名称"),
        ("特性描述","特性描述"),("进化条件","进化条件"),("蛋组","蛋组"),
        ("技能","技能"),("技能石","技能石"),("血脉技能","血脉技能"),
        ("技能学习等级","技能学习等级"),("分类","精灵分类"),
    ]:
        v = data.get(wk,"")
        if v and not v.startswith("{'fulltext'"):
            lines.append(f"{label}: {v}")

    # 描述
    for dk in ["专题描述","精灵描述","描述"]:
        d = data.get(dk,"")
        if d:
            lines.append(f"\n{d}")
            break

    # 属性克制
    if attrs:
        eff = compute_effectiveness(list(attrs))
        if eff:
            lines.append(f"\n--- 属性克制 ---\n{eff}")

    fname = name.replace("/","_").replace(":","_").replace("\\","_").replace("*","")
    fp = os.path.join(folder, f"{pid}_{fname}.txt")
    with open(fp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    # 缓存 SMW 结果，避免重复被封
    cache_file = "smw_cache.json"
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            results = json.load(f)
        print(f"Loaded {len(results)} from cache")
    else:
        results = fetch_all_pages()
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False)
        print(f"Found {len(results)} pokemon (cached)")

    stats = {}
    for i, (pn, pd) in enumerate(results.items()):
        pn = pn.replace("​","").strip()
        if not pn: continue
        smw = pd.get("printouts",{})
        wt = fetch_wikitext(pn)
        data = parse_template(wt) if wt else {}
        save_pokemon(pn, data, smw)

        po_attrs = smw.get("属性",[]) or []
        a = po_attrs[0] if po_attrs else (data.get("属性","未知").split("/")[0])
        stats[a] = stats.get(a,0)+1

        if (i+1)%50==0: print(f"  {i+1}/{len(results)}...")
        time.sleep(0.8)

    print(f"\nDone! {i+1} pokemon:")
    for c,n in sorted(stats.items(),key=lambda x:-x[1]):
        print(f"  {c}/: {n} 只")
    print(f"总计: {sum(stats.values())}")

if __name__=="__main__":
    main()
