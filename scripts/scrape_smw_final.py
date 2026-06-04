# -*- coding: utf-8 -*-
"""SMW API 批量获取全部精灵数据 → 按属性分类保存。无需逐页爬取。"""
import json, os, re, shutil, time, requests

API = "https://wiki.biligame.com/rocom/api.php"
H = {"User-Agent": "Mozilla/5.0 (Win64; x64) AppleWebKit/537.36"}
ROOT = "D:/projects/wiki"
FIELDS = ["编号","属性","生命","物攻","魔攻","物防","魔防","速度","身高","体重",
          "特性名称","特性描述","进化条件","蛋组","技能","技能石","血脉技能","图鉴描述","分类"]

TC = {
    "火":{"克":["草","虫","冰","机械"],"抗":["火","水","岩"],"被克":["水","岩","地"]},
    "水":{"克":["火","岩","地"],"抗":["水","草","电"],"被克":["草","电"]},
    "草":{"克":["水","岩","地"],"抗":["草","电","地"],"被克":["火","虫","冰","飞行"]},
    "光":{"克":["恶","幽"],"抗":["光","幻"],"被克":["草","幽"]},
    "恶":{"克":["光","幻","超能"],"抗":["恶","幽"],"被克":["光","格斗"]},
    "普通":{"克":[],"抗":[],"被克":["格斗"]},
    "电":{"克":["水","飞行"],"抗":["电"],"被克":["地"]},
    "冰":{"克":["草","地","飞行","龙"],"抗":["冰"],"被克":["火","格斗","岩","钢"]},
    "格斗":{"克":["普通","冰","岩","钢","恶"],"抗":["虫","岩"],"被克":["飞行","超能","虫"]},
    "毒":{"克":["草","虫"],"抗":["毒","草","格斗"],"被克":["地","超能"]},
    "地":{"克":["火","电","毒","岩","钢"],"抗":["岩","毒"],"被克":["水","草","冰"]},
    "飞行":{"克":["草","格斗","虫"],"抗":["地","格斗","草"],"被克":["电","冰","岩"]},
    "超能":{"克":["格斗","毒"],"抗":["超能"],"被克":["虫","幽","恶"]},
    "虫":{"克":["草","超能","恶"],"抗":["草","地","格斗"],"被克":["火","飞行","岩","毒"]},
    "岩":{"克":["火","冰","飞行","虫"],"抗":["普通","火","飞行","毒"],"被克":["水","草","格斗","地","钢"]},
    "幽":{"克":["超能","光"],"抗":["普通","幽","毒"],"被克":["恶","光"]},
    "龙":{"克":["龙"],"抗":["火","水","草","电"],"被克":["冰","龙"]},
    "钢":{"克":["冰","岩"],"抗":["普通","草","虫","岩","冰","飞行","超能","龙","钢"],"被克":["火","格斗","地"]},
    "机械":{"克":["冰","岩","钢"],"抗":["普通","草","电","飞行","超能","虫","岩","冰","龙","钢","机械"],"被克":["火","水","格斗","地"]},
    "萌":{"克":["恶","幽","格斗"],"抗":["普通","萌"],"被克":["恶","幽"]},
    "幻":{"克":["光"],"抗":["光","幻","恶"],"被克":["幽"]},
}

# 清空
if os.path.exists(ROOT):
    for item in os.listdir(ROOT):
        p = os.path.join(ROOT, item)
        if os.path.isdir(p): shutil.rmtree(p)
        elif item.endswith(".txt"): os.remove(item)


def fetch_batch(offset: int) -> dict:
    q = "[[Category:精灵]]|?" + "|?".join(FIELDS) + f"|limit=200|offset={offset}"
    r = requests.get(f"{API}?action=ask&query={q}&format=json", headers=H, timeout=30)
    if r.status_code != 200: return {}
    return r.json().get("query", {}).get("results", {})


def compute_eff(attrs: list[str]) -> str:
    weak_to, strong_vs, resist = set(), set(), set()
    for a in attrs:
        if a.strip() in TC:
            strong_vs.update(TC[a.strip()].get("克", []))
            resist.update(TC[a.strip()].get("抗", []))
            weak_to.update(TC[a.strip()].get("被克", []))
    lines = []
    if strong_vs: lines.append(f"克制: {', '.join(sorted(strong_vs))}")
    if resist: lines.append(f"抵抗: {', '.join(sorted(resist))}")
    if weak_to: lines.append(f"被克制: {', '.join(sorted(weak_to))}")
    return "\n".join(lines)


def main():
    all_data = {}
    for offset in [0, 200, 400, 600]:
        batch = fetch_batch(offset)
        all_data.update(batch)
        print(f"  offset {offset}: +{len(batch)} total {len(all_data)}")
        time.sleep(2)

    print(f"Total: {len(all_data)} entries")
    cats = {}

    import unicodedata
    def safe(s):
        result = []
        for c in str(s):
            cat = unicodedata.category(c)
            # Skip control chars, format chars (zero-width), surrogates
            if cat.startswith('C'): continue
            if c in '<>:"/\\|?*': continue
            result.append(c)
        return "".join(result).strip()

    for pn, pd in all_data.items():
        pn = pn.replace("​", "").strip()
        if not pn: continue
        po = pd.get("printouts", {})

        attrs = [safe(a) for a in (po.get("属性") or []) if a and a != "未知"]
        attrs = [a for a in attrs if a]
        primary = attrs[0] if attrs else "未知"

        pid_raw = po.get("编号") or []
        pid = safe(pid_raw[0]) if pid_raw else "???"
        pid = pid or "???"

        lines = [f"# {pn}", f"编号: {pid}", f"属性: {', '.join(attrs) if attrs else '未知'}"]
        for k in ["生命","物攻","魔攻","物防","魔防","速度"]:
            v = (po.get(k) or [None])[0]
            if v: lines.append(f"{k}: {v}")
        total = sum(int((po.get(k) or [0])[0] or 0) for k in ["生命","物攻","魔攻","物防","魔防","速度"])
        lines.append(f"种族值合计: {total}")
        for k in ["身高","体重","特性名称","特性描述","进化条件","蛋组","技能","技能石","血脉技能","图鉴描述"]:
            v = (po.get(k) or [None])[0]
            if v: lines.append(f"{k}: {v}")
        eff = compute_eff(attrs)
        if eff: lines.append(f"\n--- 属性克制 ---\n{eff}")

        folder = os.path.join(ROOT, primary)
        try:
            os.makedirs(folder, exist_ok=True)
            fname = safe(pn)[:80] or "unknown"
            fp = os.path.join(folder, f"{pid}_{fname}.txt")
            with open(fp, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            cats[primary] = cats.get(primary, 0) + 1
        except OSError as e:
            print(f"SKIP {repr(pn[:20])} folder={repr(primary[:10])} pid={repr(pid[:10])}: {e}")

    print(f"\nDone! {sum(cats.values())} files:")
    for c, n in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {c}/: {n} 只")


if __name__ == "__main__":
    main()
