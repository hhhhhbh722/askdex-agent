# -*- coding: utf-8 -*-
"""精灵图鉴规则抽取器。

第一阶段只抽确定性字段；描述、特性和进化等复杂语义留给 LLM enrich。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class KGNode:
    name: str
    type: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class KGFact:
    subject: KGNode
    predicate: str
    object: KGNode | None = None
    value_text: str | None = None
    value_number: float | None = None
    value_min: float | None = None
    value_max: float | None = None
    properties: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    extractor: str = "rule"


@dataclass(frozen=True)
class KGExtraction:
    entity: KGNode | None
    nodes: list[KGNode]
    facts: list[KGFact]
    llm_candidates: dict[str, str]


_NUMERIC_FIELDS = {
    "生命": "life",
    "物攻": "physical_attack",
    "魔攻": "magic_attack",
    "物防": "physical_defense",
    "魔防": "magic_defense",
    "速度": "speed",
}

_LIST_FIELDS = {
    "技能": ("拥有技能", "skill"),
    "血脉技能": ("拥有血脉技能", "bloodline_skill"),
    "可学技能石": ("可学技能石", "skill_stone"),
    "图鉴课题": ("图鉴课题", "task"),
    "技能解锁等级": ("技能解锁等级", "level"),
}

_ENTITY_FIELDS = {
    "主属性": ("主属性", "attribute"),
    "2属性": ("副属性", "attribute"),
    "属性": ("属性", "attribute"),
    "精灵阶段": ("精灵阶段", "stage"),
    "精灵形态": ("精灵形态", "form"),
    "分布地区": ("分布地区", "region"),
    "精灵类型": ("精灵类型", "spirit_type"),
    "地区形态名称": ("地区形态名称", "form"),
}

_TEXT_FIELDS = {
    "特性描述": "feature_description",
    "精灵描述": "spirit_description",
    "进化条件": "evolution_condition",
}


def normalize_name(value: str) -> str:
    """归一化实体名，保留中文语义但去掉模板残片和多余空白。"""
    value = re.sub(r"\[\[File:[^\]]+\]\]", "", value)
    value = re.sub(r"\{\{[^}]+\}\}", "", value)
    value = value.replace("'''", "").strip()
    value = value.strip("|=：:，,;；")
    value = re.sub(r"\s+", " ", value)
    return value


def extract_spirit_kg(text: str, filename: str = "") -> KGExtraction:
    fields = _parse_fields(text)
    name = _detect_name(text, filename, fields)
    if not name:
        return KGExtraction(entity=None, nodes=[], facts=[], llm_candidates={})

    subject = KGNode(name=name, type="spirit")
    nodes: dict[tuple[str, str], KGNode] = {
        (subject.type, _key(subject.name)): subject,
    }
    facts: list[KGFact] = []
    props: dict[str, Any] = {}

    for label, prop_name in _NUMERIC_FIELDS.items():
        value = _first(fields.get(label))
        number = _parse_float(value)
        if number is None:
            continue
        props[prop_name] = number
        facts.append(KGFact(
            subject=subject,
            predicate=label,
            value_text=value,
            value_number=number,
        ))

    for label, (predicate, target_type) in _ENTITY_FIELDS.items():
        for value in fields.get(label, []):
            for item in _split_items(value):
                if not item or item == "-" or "=" in item:
                    continue
                node = KGNode(name=item, type=target_type)
                node = _remember(nodes, node)
                facts.append(KGFact(subject=subject, predicate=predicate, object=node))

    for label, (predicate, target_type) in _LIST_FIELDS.items():
        for value in fields.get(label, []):
            for item in _split_items(value):
                if not item or item == "-" or "=" in item:
                    continue
                node = KGNode(name=item, type=target_type)
                node = _remember(nodes, node)
                facts.append(KGFact(subject=subject, predicate=predicate, object=node))

    for label, predicate in (("体型", "体型"), ("重量", "重量")):
        value = _first(fields.get(label))
        if not value:
            continue
        value_min, value_max = _parse_range(value)
        facts.append(KGFact(
            subject=subject,
            predicate=predicate,
            value_text=value,
            value_min=value_min,
            value_max=value_max,
        ))
        props[predicate] = value

    for label in ("是否有异色", "更新版本", "宠物立绘形态"):
        value = _first(fields.get(label))
        if value:
            facts.append(KGFact(subject=subject, predicate=label, value_text=value))
            props[label] = value

    if props:
        subject = KGNode(name=subject.name, type=subject.type, properties=props)
        nodes[(subject.type, _key(subject.name))] = subject
        facts = [
            KGFact(
                subject=subject,
                predicate=f.predicate,
                object=f.object,
                value_text=f.value_text,
                value_number=f.value_number,
                value_min=f.value_min,
                value_max=f.value_max,
                properties=f.properties,
                confidence=f.confidence,
                extractor=f.extractor,
            )
            for f in facts
        ]

    llm_candidates = {
        prop: "\n".join(v for v in fields.get(label, []) if v).strip()
        for label, prop in _TEXT_FIELDS.items()
        if fields.get(label)
    }
    return KGExtraction(
        entity=subject,
        nodes=list(nodes.values()),
        facts=_dedupe_facts(facts),
        llm_candidates=llm_candidates,
    )


def _parse_fields(text: str) -> dict[str, list[str]]:
    fields: dict[str, list[str]] = {}
    current: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("<") or line.startswith("[[File:"):
            continue
        match = re.match(r"^([^:：]{1,24})[:：]\s*(.*)$", line)
        if match:
            key = normalize_name(match.group(1))
            raw_value = match.group(2).strip()
            value = normalize_name(raw_value)
            if key.startswith("分类"):
                continue
            if raw_value.startswith("|") and "=" in raw_value:
                for part in raw_value.split("|"):
                    if "=" not in part:
                        continue
                    extra_key, extra_value = part.split("=", 1)
                    extra_key = normalize_name(extra_key)
                    extra_value = normalize_name(extra_value)
                    if extra_key and extra_value:
                        fields.setdefault(extra_key, []).append(extra_value)
            else:
                fields.setdefault(key, []).append(value)
            current = key
            continue
        if current and current in {"精灵描述", "特性描述", "进化条件", "图鉴课题"}:
            fields[current][-1] = normalize_name(f"{fields[current][-1]}\n{line}")

    # Some wiki fragments look like "2属性: |特性=契约的形状".
    for key, values in list(fields.items()):
        cleaned: list[str] = []
        for value in values:
            if "|" in value and "=" in value:
                first, *rest = value.split("|")
                if first.strip():
                    cleaned.append(normalize_name(first))
                for part in rest:
                    if "=" not in part:
                        continue
                    extra_key, extra_value = part.split("=", 1)
                    extra_key = normalize_name(extra_key)
                    extra_value = normalize_name(extra_value)
                    if extra_key and extra_value:
                        fields.setdefault(extra_key, []).append(extra_value)
            else:
                cleaned.append(value)
        fields[key] = cleaned
    return fields


def _detect_name(text: str, filename: str, fields: dict[str, list[str]]) -> str:
    name = _first(fields.get("精灵名称"))
    if name:
        return name
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#"):
            return normalize_name(line.lstrip("#"))
    if filename:
        cleaned = re.sub(r"^[0-9a-fA-F-]{36}_", "", filename)
        cleaned = re.sub(r"\.[^.]+$", "", cleaned)
        return normalize_name(cleaned)
    return ""


def _split_items(value: str) -> list[str]:
    value = normalize_name(value)
    value = value.replace("、", ",").replace("，", ",").replace("；", ",")
    return [normalize_name(item) for item in value.split(",") if normalize_name(item)]


def _parse_float(value: str | None) -> float | None:
    if not value:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", value)
    if not match:
        return None
    return float(match.group(0))


def _parse_range(value: str | None) -> tuple[float | None, float | None]:
    if not value:
        return None, None
    nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", value)]
    if not nums:
        return None, None
    if len(nums) == 1:
        return nums[0], nums[0]
    return min(nums[0], nums[1]), max(nums[0], nums[1])


def _first(values: list[str] | None) -> str:
    if not values:
        return ""
    return next((v for v in values if v), "")


def _key(value: str) -> str:
    return normalize_name(value).lower()


def _remember(nodes: dict[tuple[str, str], KGNode], node: KGNode) -> KGNode:
    key = (node.type, _key(node.name))
    if key not in nodes:
        nodes[key] = node
    return nodes[key]


def _dedupe_facts(facts: list[KGFact]) -> list[KGFact]:
    seen: set[tuple[Any, ...]] = set()
    result: list[KGFact] = []
    for fact in facts:
        marker = (
            fact.subject.type,
            _key(fact.subject.name),
            fact.predicate,
            fact.object.type if fact.object else "",
            _key(fact.object.name) if fact.object else "",
            fact.value_text or "",
            fact.value_number,
            fact.value_min,
            fact.value_max,
        )
        if marker in seen:
            continue
        seen.add(marker)
        result.append(fact)
    return result
