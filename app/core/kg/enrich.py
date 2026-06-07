# -*- coding: utf-8 -*-
"""LLM enrichment for rule-extracted KG candidates."""
from __future__ import annotations

import json
import re
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.kg.extractor import KGFact, KGNode, extract_spirit_kg, normalize_name
from app.core.kg.repository import add_relation, upsert_entity
from app.infrastructure.database.models import Document, DocumentChunk


async def enrich_document_with_llm(
    session: AsyncSession,
    document: Document,
    llm: Any,
    max_relations: int = 12,
) -> dict[str, Any]:
    chunks_r = await session.execute(
        select(DocumentChunk)
        .where(DocumentChunk.document_id == document.id)
        .order_by(DocumentChunk.chunk_index.asc())
    )
    chunks = chunks_r.scalars().all()
    text = "\n".join(chunk.content for chunk in chunks)
    extraction = extract_spirit_kg(text, filename=document.filename)
    if not extraction.entity or not extraction.llm_candidates:
        return {"document_id": document.id, "filename": document.filename, "relations": 0, "skipped": True}

    prompt = _build_enrich_prompt(extraction.entity.name, extraction.llm_candidates, max_relations)
    raw = await llm.acomplete([
        {"role": "system", "content": "你是知识图谱抽取器，只输出合法 JSON。"},
        {"role": "user", "content": prompt},
    ], temperature=0.0)
    payload = _parse_json_object(raw)
    items = payload.get("relations", []) if isinstance(payload, dict) else []

    subject = await upsert_entity(session, extraction.entity, source_document_id=document.id)
    count = 0
    for item in items[:max_relations]:
        if not isinstance(item, dict):
            continue
        predicate = normalize_name(str(item.get("predicate") or ""))
        target = normalize_name(str(item.get("object") or ""))
        target_type = normalize_name(str(item.get("object_type") or "concept")) or "concept"
        evidence = str(item.get("evidence") or "")[:500]
        confidence = _safe_float(item.get("confidence"), default=0.65)
        if not predicate or not target:
            continue
        obj = await upsert_entity(
            session,
            KGNode(name=target, type=target_type),
            source_document_id=document.id,
        )
        await add_relation(
            session=session,
            fact=KGFact(
                subject=extraction.entity,
                predicate=predicate,
                object=KGNode(name=target, type=target_type),
                properties={"evidence": evidence},
                confidence=max(0.0, min(confidence, 1.0)),
                extractor="llm",
            ),
            subject_id=subject.id,
            object_id=obj.id,
            source_document_id=document.id,
            source_chunk_id=chunks[0].id if chunks else None,
        )
        count += 1
    return {"document_id": document.id, "filename": document.filename, "relations": count, "raw": payload}


def _build_enrich_prompt(name: str, candidates: dict[str, str], max_relations: int) -> str:
    body = json.dumps(candidates, ensure_ascii=False, indent=2)
    return f"""请从下面材料中抽取“{name}”的复杂知识图谱关系。
只抽取明确能从原文支持的关系，最多 {max_relations} 条。
重点关系类型：进化关系、形态变化关系、设定关系、特性效果关系、技能效果关系。

输出 JSON 格式：
{{
  "relations": [
    {{
      "predicate": "关系名称",
      "object": "客体实体或概念",
      "object_type": "spirit|form|condition|effect|skill|concept",
      "confidence": 0.0,
      "evidence": "原文证据"
    }}
  ]
}}

材料：
{body}
"""


def _parse_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            return {"relations": [], "error": "llm_json_parse_failed", "raw": raw[:1000]}
        try:
            return json.loads(match.group(0))
        except Exception:
            return {"relations": [], "error": "llm_json_parse_failed", "raw": raw[:1000]}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default
