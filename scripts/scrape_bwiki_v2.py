# -*- coding: utf-8 -*-
"""爬取 BiliGame 洛克王国 Wiki 精灵详情页 → 按属性分类保存完整数据。"""
import json
import os
import re
import time
import shutil
import requests

BASE_API = "https://wiki.biligame.com/rocom/api.php"
HEADERS = {"User-Agent": "Mozilla/5.0 (Win64; x64) AppleWebKit/537.36"}
OUTPUT_ROOT = "D:/projects/wiki"

# 清空旧数据
if os.path.exists(OUTPUT_ROOT):
    for item in os.listdir(OUTPUT_ROOT):
        item_path = os.path.join(OUTPUT_ROOT, item)
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)
        elif item.endswith(".txt"):
            os.remove(item_path)


def fetch_all_pages():
    """SMW API: 获取所有精灵的页面名和属性。"""
    query = "[[Category:精灵]]|?" + "|?".join([
        "编号", "属性", "HP", "物攻", "魔攻", "物防", "魔防", "速度",
        "身高", "体重", "进化条件", "特性名称", "特性描述", "蛋组",
        "技能", "技能石", "血脉技能", "图鉴描述", "分类",
    ]) + "|limit=800"
    url = f"{BASE_API}?action=ask&query={query}&format=json"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    data = resp.json()
    results = data.get("query", {}).get("results", {})
    print(f"SMW returned {len(results)} pokemon")
    return results


def fetch_wikitext(page_name: str) -> str:
    """获取页面 wikitext 源码。"""
    params = {"action": "parse", "page": page_name, "prop": "wikitext", "format": "json"}
    try:
        resp = requests.get(BASE_API, params=params, headers=HEADERS, timeout=15)
        data = resp.json()
        return data.get("parse", {}).get("wikitext", {}).get("*", "")
    except Exception:
        return ""


def parse_template(wikitext: str) -> dict:
    """从 wikitext 的 {{宠物信息/精灵|...}} 模板提取所有参数。"""
    info = {}
    # 匹配模板块
    m = re.search(r'\{\{宠物信息/精灵[\s\S]*?\n\}\}', wikitext)
    if not m:
        m = re.search(r'\{\{宠物信息\s*\|[\s\S]*?\n\}\}', wikitext)
    if not m:
        return info

    block = m.group(0)
    # 解析 |key=value 对
    pattern = re.compile(r'\|\s*([^=|\n]+?)\s*=\s*(.*?)(?=\n\||\n\}\})', re.S)
    for match in pattern.finditer(block):
        key = match.group(1).strip()
        value = match.group(2).strip()
        # 清理 HTML 标签
        value = re.sub(r'<[^>]+>', '', value)
        value = re.sub(r'\{\{[^}]+\}\}', '', value)
        info[key] = value
    return info


def save_pokemon(name: str, data: dict, smw_info: dict):
    """保存单个精灵文档到属性分类文件夹。"""
    # 属性
    attrs = set()
    for k in ["属性", "属性1", "属性2"]:
        v = data.get(k, "") or ""
        for a in re.split(r'[/,，/]', v):
            a = a.strip()
            if a:
                attrs.add(a)

    # SMW 中的属性
    smw_attrs = smw_info.get("属性", [])
    for a in smw_attrs:
        if a and a != "未知":
            attrs.add(a)

    primary = list(attrs)[0] if attrs else "未知"
    category = primary

    folder = os.path.join(OUTPUT_ROOT, category)
    os.makedirs(folder, exist_ok=True)

    # 编号
    pid = data.get("编号", data.get("编号名称", ""))
    if not pid:
        pid = smw_info.get("编号", [name])[0] if smw_info.get("编号") else name

    # 构建文本
    lines = [f"# {name}"]
    if pid:
        lines.append(f"编号: {pid}")

    # wikitext 中文键名 → 显示标签
    KEY_MAP = [
        ("编号名称", "编号"), ("编号", "编号"),
        ("属性", "属性"), ("分类", "分类"),
        ("身高", "身高"), ("体重", "体重"),
        ("生命", "HP"), ("HP", "HP"),
        ("物攻", "物攻"), ("魔攻", "魔攻"),
        ("物防", "物防"), ("魔防", "魔防"),
        ("速度", "速度"),
        ("特性名称", "特性名称"), ("特性描述", "特性描述"),
        ("进化条件", "进化条件"), ("蛋组", "蛋组"),
        ("技能", "技能"), ("技能石", "技能石"),
        ("血脉技能", "血脉技能"), ("技能学习等级", "技能学习等级"),
        ("专题描述", "图鉴描述"), ("图鉴描述", "图鉴描述"),
        ("分布位置", "分布位置"), ("精灵形态", "精灵形态"),
    ]
    seen_labels = set()
    for wk_key, label in KEY_MAP:
        if label in seen_labels:
            continue
        v = data.get(wk_key, "") or ""
        if not v:
            # 试试 SMW
            smw_val = smw_info.get(wk_key, []) or smw_info.get(("_".join(wk_key.split())).strip(), [])
            if smw_val:
                v = str(smw_val[0])
        if v:
            lines.append(f"{label}: {v}")
            seen_labels.add(label)

    # 描述
    for dk in ["专题描述", "描述", "精灵描述"]:
        desc = data.get(dk, "")
        if desc:
            lines.append(f"\n{desc}")
            break

    # 种族值合计
    try:
        total = sum(int(data.get(k, 0) or 0) for k in ["生命", "物攻", "魔攻", "物防", "魔防", "速度"])
    except (ValueError, TypeError):
        total = 0
    if total > 0:
        lines.append(f"\n种族值合计: {total}")

    # 写入文件
    safe_name = name.replace("/", "_").replace(":", "_").replace("\\", "_").replace("*", "")
    filename = f"{pid}_{safe_name}.txt" if pid else f"{safe_name}.txt"
    filepath = os.path.join(folder, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    print("Step 1: Fetching all pokemon from SMW API...")
    smw_results = fetch_all_pages()

    # 按属性分类统计
    attr_count = {}

    print("Step 2: Fetching detail pages...")
    for i, (page_name, page_data) in enumerate(smw_results.items()):
        # 清理零宽空格
        clean_name = page_name.replace("​", "").strip()
        if not clean_name:
            continue

        prints = page_data.get("printouts", {})

        # 先尝试 SMW 数据
        smw_attrs = prints.get("属性", [])
        smw_hp = prints.get("HP", [])
        # 如果有 SMW 基础数据就先用
        smw_info = {k: v for k, v in prints.items() if v}

        # 获取完整 wikitext
        wt = fetch_wikitext(clean_name)
        data = parse_template(wt) if wt else {}

        # 合并数据
        save_pokemon(clean_name, data, smw_info)

        # 统计
        attrs = set()
        for a in smw_attrs:
            attrs.add(a)
        if not attrs and data.get("属性"):
            for a in re.split(r'[/,，/]', data["属性"]):
                if a.strip():
                    attrs.add(a.strip())
        cat = list(attrs)[0] if attrs else "未知"
        attr_count[cat] = attr_count.get(cat, 0) + 1

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(smw_results)}...")
        time.sleep(1.0)  # 慢速避免被 WAF 拦截

    print(f"\nDone! {i + 1} pokemon saved to {OUTPUT_ROOT}:")
    for cat, count in sorted(attr_count.items(), key=lambda x: -x[1]):
        print(f"  {cat}/: {count} 只")
    print(f"\n总计: {sum(attr_count.values())} 只")


if __name__ == "__main__":
    main()
