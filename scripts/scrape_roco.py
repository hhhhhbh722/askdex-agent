# -*- coding: utf-8 -*-
"""爬取洛克王国 wiki 精灵数据 → 生成知识库 TXT 文件。"""
import json
import os
import time
import requests

BASE_URL = "https://wiki.lcx.cab/lk/get_pokemon_data.php"
OUTPUT_DIR = "kb_roco_docs"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://wiki.lcx.cab/lk/tujian.php",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

page = 1
total = 0
seen = set()

while True:
    url = f"{BASE_URL}?page={page}&exclude_details=1"
    print(f"Fetching page {page}...", end=" ")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        data = resp.json()
    except Exception as e:
        print(f"Error: {e}")
        break

    if not data or not isinstance(data, list) or "error" in str(data):
        print("Done (no more data)")
        break

    count_this_page = 0
    for mon in data:
        mon_id = mon.get("t_id", "unknown")
        name = mon.get("name", "unknown")
        form = mon.get("form_display_name", "")
        key = f"{mon_id}_{form}" if form else mon_id

        if key in seen:
            continue
        seen.add(key)

        # 生成可读文本
        display = form if form else name
        lines = [
            f"# {display}",
            f"编号: {mon_id}",
            f"属性: {mon.get('attributes', '')}",
            f"身高: {mon.get('height', '')}",
            f"体重: {mon.get('weight', '')}kg",
            f"HP: {mon.get('hp', '')} | 攻击: {mon.get('attack', '')} | 特攻: {mon.get('special_attack', '')}",
            f"防御: {mon.get('defense', '')} | 特防: {mon.get('special_defense', '')} | 速度: {mon.get('speed', '')}",
            f"特性: {mon.get('abilities_text', '')}",
            f"进化阶段: {mon.get('evolution_stage', '')}",
            f"进化条件: {mon.get('evolution_condition', '')}",
            f"蛋组: {mon.get('egg_group', '')}",
            f"蛋直径: {mon.get('egg_diameter', '')} | 蛋重量: {mon.get('egg_weight', '')}",
            f"技能: {mon.get('moves', '')}",
            f"技能石: {mon.get('jinengshi', '')}",
            f"血脉: {mon.get('xuemai', '')}",
        ]
        if mon.get("description"):
            lines.append(f"图鉴描述: {mon['description']}")

        filename = f"{mon_id}_{display}.txt".replace("/", "_").replace(":", "_")
        filepath = os.path.join(OUTPUT_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        count_this_page += 1
        total += 1

    print(f"→ saved {count_this_page} new (total: {total})")
    if count_this_page == 0:
        print("No new entries on this page, stopping")
        break

    page += 1
    time.sleep(0.2)  # 礼貌等待

print(f"\nDone! {total} pokemon saved to '{OUTPUT_DIR}/'")
