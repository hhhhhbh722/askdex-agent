# -*- coding: utf-8 -*-
"""基于爬取的精灵数据自动生成 QA 评测对。"""
import json
import os
import random
from pathlib import Path

DOCS_DIR = "kb_roco_docs"
OUTPUT = "qa_pairs.json"

random.seed(42)
docs = []
for fname in os.listdir(DOCS_DIR):
    if fname.endswith(".txt"):
        with open(os.path.join(DOCS_DIR, fname), encoding="utf-8") as f:
            docs.append({"file": fname, "text": f.read()})

print(f"Loaded {len(docs)} documents")

qa_pairs = []

# 对每个文档生成一个问题
for doc in docs:
    lines = doc["text"].strip().split("\n")
    data = {}
    for line in lines:
        if ": " in line:
            k, v = line.split(": ", 1)
            data[k.strip()] = v.strip()

    name = data.get("# " + doc["file"].split("_", 1)[1].replace(".txt", "")) or doc["file"]
    pid = data.get("编号", "")
    attrs = data.get("属性", "")
    hp = data.get("HP", "")
    atk = data.get("攻击", "")
    spd = data.get("速度", "")
    evo_cond = data.get("进化条件", "")
    ability = data.get("特性", "")
    moves = data.get("技能", "")
    egg = data.get("蛋组", "")

    # 根据数据完整性生成不同问题
    questions = []

    # 基础属性问题
    if pid and name:
        questions.append({
            "question": f"编号{pid}的精灵是什么？",
            "answer_ref": name,
            "type": "factual",
        })
        questions.append({
            "question": f"{name}的属性是什么？" if attrs else "",
            "answer_ref": attrs,
            "type": "factual",
        })

    # 种族值问题
    if hp:
        questions.append({
            "question": f"{name}的HP种族值是多少？",
            "answer_ref": hp,
            "type": "numerical",
        })
    if atk:
        questions.append({
            "question": f"{name}的攻击种族值是多少？",
            "answer_ref": atk,
            "type": "numerical",
        })
    if spd:
        questions.append({
            "question": f"{name}的速度是多少？",
            "answer_ref": spd,
            "type": "numerical",
        })

    # 进化问题
    if evo_cond and evo_cond not in ("主线赠送", "御三家", "-"):
        questions.append({
            "question": f"{name}的进化条件是什么？",
            "answer_ref": evo_cond,
            "type": "factual",
        })

    # 特性/技能问题
    if ability and len(ability) > 5:
        questions.append({
            "question": f"{name}有什么特性？",
            "answer_ref": ability[:100],
            "type": "descriptive",
        })

    # 过滤空问题
    questions = [q for q in questions if q["question"]]

    # 每只最多取3个问题
    for q in questions[:3]:
        q["source_file"] = doc["file"]
        qa_pairs.append(q)

random.shuffle(qa_pairs)
print(f"Generated {len(qa_pairs)} QA pairs")

# 输出前100条作为评测集
eval_set = qa_pairs[:100]
with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(eval_set, f, ensure_ascii=False, indent=2)

print(f"Saved {len(eval_set)} QA pairs to {OUTPUT}")
print("\nSample:")
for q in eval_set[:5]:
    print(f"  Q: {q['question']}")
    print(f"  A: {q['answer_ref'][:80]}")
    print()
