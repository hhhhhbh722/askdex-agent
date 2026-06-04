# -*- coding: utf-8 -*-
"""爬取 BiliGame 洛克王国 Wiki 精灵图鉴 → 按属性分类保存。"""
import json
import os
import time
import urllib.parse
import requests

BASE = "https://wiki.biligame.com/rocom/api.php"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
OUTPUT_ROOT = "D:/projects/wiki"

# 属性英文→中文映射
ATTR_MAP = {
    "fire": "火", "water": "水", "grass": "草", "light": "光",
    "dark": "恶", "normal": "普通", "electric": "电", "ice": "冰",
    "fighting": "格斗", "poison": "毒", "ground": "地面", "flying": "飞行",
    "psychic": "超能", "bug": "虫", "rock": "岩石", "ghost": "幽灵",
    "dragon": "龙", "steel": "钢", "fairy": "妖精",
    "fire_fighting": "火+格斗", "water_dark": "水+恶",
    "首领": "首领", "mechanical": "机械",
}


def fetch_all_pokemon():
    """获取所有精灵列表（含属性）。"""
    query = "[[Category:精灵]]|?属性|?分类|?攻击|?防御|?HP|?特攻|?特防|?速度|?进化条件|?蛋组|limit=600"
    encoded = urllib.parse.quote(query)
    url = f"{BASE}?action=ask&query={encoded}&format=json"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    data = resp.json()
    return data.get("query", {}).get("results", {})


def fetch_detail(page_name: str) -> dict:
    """获取单个精灵页面 wikitext。"""
    params = {
        "action": "parse",
        "page": page_name,
        "prop": "wikitext",
        "format": "json",
    }
    try:
        resp = requests.get(BASE, params=params, headers=HEADERS, timeout=15)
        data = resp.json()
        return data.get("parse", {}).get("wikitext", {})
    except Exception:
        return {}


def parse_wikitext(text: str, page_name: str) -> dict:
    """从 wikitext 提取结构化信息。"""
    info = {"name": page_name}
    # 提取模板参数 {{精灵信息|参数1=值1|参数2=值2}}
    import re
    # 找 {{精灵信息 或 {{精灵
    match = re.search(r'\{\{(?:精灵信息|宠物信息)\s*\|([^}]+)\}\}', text, re.S)
    if not match:
        match = re.search(r'\{\{精灵\s*\|([^}]+)\}\}', text, re.S)
    if match:
        params_str = match.group(1)
        # 解析键值对
        for param in params_str.split("|"):
            if "=" in param:
                k, v = param.split("=", 1)
                info[k.strip()] = v.strip()
    return info


def save_pokemon(pokemon_list: list):
    """按属性分类保存精灵文档。"""
    stats = {}
    for name, data in pokemon_list:
        prints = data.get("printouts", {})
        attrs = prints.get("属性", [])
        attr_str = attrs[0] if attrs else "未知"
        category = attr_str.replace(" ", "").replace("/", "_")

        if category not in stats:
            stats[category] = 0
        stats[category] += 1

        # 创建属性文件夹
        folder = os.path.join(OUTPUT_ROOT, category)
        os.makedirs(folder, exist_ok=True)

        # 生成文本
        lines = [f"# {name}"]
        for field in ["攻击", "防御", "HP", "特攻", "特防", "速度", "属性", "分类", "进化条件", "蛋组"]:
            val = prints.get(field, [])
            if val:
                lines.append(f"{field}: {val[0]}")

        filename = name.replace("/", "_").replace(":", "_").replace("\\", "_")
        filepath = os.path.join(folder, f"{filename}.txt")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    return stats


def main():
    print("Fetching all pokemon from BiliGame Wiki...")
    results = fetch_all_pokemon()
    print(f"Found {len(results)} pokemon")

    pokemon_list = list(results.items())

    stats = save_pokemon(pokemon_list)

    print(f"\nSaved to {OUTPUT_ROOT}:")
    for cat, count in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {cat}/: {count} 只")

    print(f"\n总计: {sum(stats.values())} 只")
    print("Done!")


if __name__ == "__main__":
    main()
