# -*- coding: utf-8 -*-
"""轻量知识图谱存取与检索服务。"""
from __future__ import annotations

import json
import re
from dataclasses import asdict
from typing import Any

from sqlalchemy import delete, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.kg.extractor import KGExtraction, KGFact, KGNode, extract_spirit_kg, normalize_name
from app.infrastructure.database.models import Document, DocumentChunk, KGEntity, KGRelation

_ENTITY_STOPWORDS = {"精灵", "技能", "属性", "图鉴", "知识", "资料", "哪些", "什么", "一下"}
_CORE_PREDICATE_BONUS = {
    "主属性": 0.4,
    "副属性": 0.35,
    "属性": 0.3,
    "拥有技能": 0.45,
    "拥有血脉技能": 0.35,
    "可学技能石": 0.25,
    "特性效果": 0.45,
    "技能效果": 0.4,
    "进化条件": 0.45,
    "形态变化": 0.35,
}


def is_spirit_document(document: Document) -> bool:
    meta = document.meta or {}
    group = str(meta.get("group") or "")
    parent = str(meta.get("parent_group") or "")
    return parent == "精灵图鉴" or group == "精灵图鉴" or group.startswith("精灵图鉴 /")


async def clear_kg(session: AsyncSession) -> None:
    await session.execute(delete(KGRelation))
    await session.execute(delete(KGEntity))


async def build_document_kg(session: AsyncSession, document: Document) -> dict[str, Any]:
    chunks_r = await session.execute(
        select(DocumentChunk)
        .where(DocumentChunk.document_id == document.id)
        .order_by(DocumentChunk.chunk_index.asc())
    )
    chunks = chunks_r.scalars().all()
    text = "\n".join(chunk.content for chunk in chunks)
    extraction = extract_spirit_kg(text, filename=document.filename)
    if not extraction.entity:
        return {"document_id": document.id, "filename": document.filename, "entities": 0, "relations": 0}

    entity_map: dict[tuple[str, str], KGEntity] = {}
    for node in extraction.nodes:
        entity_map[(node.type, _key(node.name))] = await upsert_entity(
            session=session,
            node=node,
            source_document_id=document.id,
        )

    first_chunk_id = chunks[0].id if chunks else None
    relation_count = 0
    for fact in extraction.facts:
        subject = entity_map[(fact.subject.type, _key(fact.subject.name))]
        obj = None
        if fact.object:
            obj = entity_map[(fact.object.type, _key(fact.object.name))]
        await add_relation(
            session=session,
            fact=fact,
            subject_id=subject.id,
            object_id=obj.id if obj else None,
            source_document_id=document.id,
            source_chunk_id=first_chunk_id,
        )
        relation_count += 1

    if extraction.llm_candidates:
        subject = entity_map[(extraction.entity.type, _key(extraction.entity.name))]
        props = dict(subject.properties or {})
        props["llm_candidates"] = extraction.llm_candidates
        subject.properties = props

    return {
        "document_id": document.id,
        "filename": document.filename,
        "spirit": extraction.entity.name,
        "entities": len(extraction.nodes),
        "relations": relation_count,
        "llm_candidate_fields": list(extraction.llm_candidates),
    }


async def upsert_entity(session: AsyncSession, node: KGNode, source_document_id: str | None = None) -> KGEntity:
    normalized = _key(node.name)
    r = await session.execute(
        select(KGEntity).where(KGEntity.type == node.type, KGEntity.normalized_name == normalized)
    )
    entity = r.scalar_one_or_none()
    if entity:
        props = dict(entity.properties or {})
        props.update(node.properties or {})
        entity.name = node.name
        entity.properties = props
        if source_document_id and not entity.source_document_id:
            entity.source_document_id = source_document_id
        return entity

    entity = KGEntity(
        name=node.name,
        normalized_name=normalized,
        type=node.type,
        properties=node.properties or {},
        source_document_id=source_document_id,
    )
    session.add(entity)
    await session.flush()
    return entity


async def add_relation(
    session: AsyncSession,
    fact: KGFact,
    subject_id: str,
    object_id: str | None,
    source_document_id: str | None,
    source_chunk_id: str | None,
) -> KGRelation:
    relation = KGRelation(
        subject_id=subject_id,
        predicate=fact.predicate,
        object_id=object_id,
        value_text=fact.value_text,
        value_number=fact.value_number,
        value_min=fact.value_min,
        value_max=fact.value_max,
        properties=fact.properties or {},
        source_document_id=source_document_id,
        source_chunk_id=source_chunk_id,
        confidence=fact.confidence,
        extractor=fact.extractor,
    )
    session.add(relation)
    await session.flush()
    return relation


async def kg_stats(session: AsyncSession) -> dict[str, Any]:
    entity_count = await session.scalar(select(func.count()).select_from(KGEntity))
    relation_count = await session.scalar(select(func.count()).select_from(KGRelation))
    by_type_r = await session.execute(
        select(KGEntity.type, func.count()).group_by(KGEntity.type).order_by(KGEntity.type)
    )
    by_predicate_r = await session.execute(
        select(KGRelation.predicate, func.count()).group_by(KGRelation.predicate).order_by(KGRelation.predicate)
    )
    return {
        "entities": int(entity_count or 0),
        "relations": int(relation_count or 0),
        "entity_types": {row[0]: int(row[1]) for row in by_type_r.all()},
        "predicates": {row[0]: int(row[1]) for row in by_predicate_r.all()},
    }


async def search_entities(
    session: AsyncSession,
    q: str,
    entity_type: str = "",
    limit: int = 20,
) -> list[dict[str, Any]]:
    qn = _key(q)
    stmt = select(KGEntity)
    if qn:
        like = f"%{qn}%"
        stmt = stmt.where(or_(KGEntity.normalized_name.ilike(like), KGEntity.name.ilike(f"%{q}%")))
    if entity_type:
        stmt = stmt.where(KGEntity.type == entity_type)
    stmt = stmt.order_by(KGEntity.type.asc(), KGEntity.name.asc()).limit(limit)
    r = await session.execute(stmt)
    return [_entity_dict(e) for e in r.scalars().all()]


async def entity_relations(
    session: AsyncSession,
    entity: str,
    depth: int = 1,
    limit: int = 120,
) -> dict[str, Any]:
    root = await _find_entity(session, entity)
    if not root:
        return {"nodes": [], "edges": [], "count": 0}

    depth = max(1, min(depth, 2))
    nodes: dict[str, dict[str, Any]] = {root.id: _entity_dict(root)}
    edges: list[dict[str, Any]] = []
    frontier = [root.id]
    seen_edges: set[str] = set()

    for _ in range(depth):
        if not frontier or len(edges) >= limit:
            break
        rels_r = await session.execute(
            select(KGRelation).where(
                or_(KGRelation.subject_id.in_(frontier), KGRelation.object_id.in_(frontier))
            ).limit(limit - len(edges))
        )
        next_frontier: list[str] = []
        for rel in rels_r.scalars().all():
            if rel.id in seen_edges:
                continue
            seen_edges.add(rel.id)
            subject = await session.get(KGEntity, rel.subject_id)
            obj = await session.get(KGEntity, rel.object_id) if rel.object_id else None
            if subject:
                nodes[subject.id] = _entity_dict(subject)
            if obj:
                nodes[obj.id] = _entity_dict(obj)
                next_frontier.append(obj.id)
            value_id = ""
            if not obj and rel.value_text:
                value_id = f"value:{rel.id}"
                nodes[value_id] = {"id": value_id, "name": rel.value_text, "type": "value", "properties": {}}
            target = obj.id if obj else value_id
            if subject and target:
                edges.append({
                    "id": rel.id,
                    "source": subject.id,
                    "target": target,
                    "label": rel.predicate,
                    "predicate": rel.predicate,
                    "confidence": rel.confidence,
                    "extractor": rel.extractor,
                })
            if subject and subject.id not in frontier:
                next_frontier.append(subject.id)
            if len(edges) >= limit:
                break
        frontier = list(dict.fromkeys(next_frontier))

    return {"nodes": list(nodes.values()), "edges": edges, "count": len(edges)}


async def graph_search(session: AsyncSession, q: str, limit: int = 20) -> dict[str, Any]:
    entities = await search_entities(session, q=q, limit=limit)
    if entities:
        return {"query": q, "mode": "entity", "entities": entities, "relations": []}
    rels_r = await session.execute(
        select(KGRelation)
        .where(KGRelation.predicate.ilike(f"%{q}%"))
        .limit(limit)
    )
    relations = []
    for rel in rels_r.scalars().all():
        subject = await session.get(KGEntity, rel.subject_id)
        obj = await session.get(KGEntity, rel.object_id) if rel.object_id else None
        relations.append(_relation_dict(rel, subject, obj))
    return {"query": q, "mode": "relation", "entities": [], "relations": relations}


async def retrieve_kg_context(session: AsyncSession, query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """从 KG 召回候选文档/事实，不依赖意图识别。

    策略：
    - 命中实体名：返回该实体的一跳关系和来源 chunk；
    - 命中技能/属性等客体：返回指向它的精灵关系；
    - 命中谓词：返回相关关系。
    """
    q = query.strip()
    if not q:
        return []

    entities = await _match_entities_for_query(session, q, limit=max(top_k * 6, 30))
    relations: list[KGRelation] = []
    seen_rel_ids: set[str] = set()

    for entity in entities:
        rels_r = await session.execute(
            select(KGRelation)
            .where(or_(KGRelation.subject_id == entity.id, KGRelation.object_id == entity.id))
            .limit(80)
        )
        for rel in rels_r.scalars().all():
            if rel.id not in seen_rel_ids:
                relations.append(rel)
                seen_rel_ids.add(rel.id)

    # Query may directly mention relation names such as "主属性" or "速度".
    predicate_terms = _query_terms(q)
    if predicate_terms:
        clauses = [KGRelation.predicate.ilike(f"%{term}%") for term in predicate_terms if len(term) >= 2]
        if clauses:
            rels_r = await session.execute(select(KGRelation).where(or_(*clauses)).limit(80))
            for rel in rels_r.scalars().all():
                if rel.id not in seen_rel_ids:
                    relations.append(rel)
                    seen_rel_ids.add(rel.id)

    if not relations and not entities:
        return []

    query_terms = _query_terms(q)
    doc_scores: dict[str, float] = {}
    doc_facts: dict[str, list[str]] = {}
    for rank, rel in enumerate(relations):
        subject = await session.get(KGEntity, rel.subject_id)
        obj = await session.get(KGEntity, rel.object_id) if rel.object_id else None
        doc_id = rel.source_document_id or (subject.source_document_id if subject else None)
        if not doc_id:
            continue
        fact_text = _format_fact(subject, rel, obj)
        doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + _relation_relevance(
            query=q,
            terms=query_terms,
            rank=rank,
            subject=subject,
            rel=rel,
            obj=obj,
        )
        doc_facts.setdefault(doc_id, [])
        if fact_text and fact_text not in doc_facts[doc_id]:
            doc_facts[doc_id].append(fact_text)

    for entity in entities:
        if entity.source_document_id:
            entity_bonus = 1.5 if entity.name in q else 0.6
            doc_scores[entity.source_document_id] = doc_scores.get(entity.source_document_id, 0.0) + entity_bonus
            doc_facts.setdefault(entity.source_document_id, []).append(f"命中实体：{entity.name}（{entity.type}）")

    ranked_doc_ids = sorted(doc_scores, key=lambda did: doc_scores[did], reverse=True)[:top_k]
    results: list[dict[str, Any]] = []
    for rank, doc_id in enumerate(ranked_doc_ids):
        chunks_r = await session.execute(
            select(DocumentChunk)
            .where(DocumentChunk.document_id == doc_id)
            .order_by(DocumentChunk.chunk_index.asc())
            .limit(3)
        )
        chunks = chunks_r.scalars().all()
        doc = await session.get(Document, doc_id)
        facts = doc_facts.get(doc_id, [])[:24]
        chunk_text = "\n".join(chunk.content[:1200] for chunk in chunks)
        content_parts = []
        if facts:
            content_parts.append("## 知识图谱命中事实\n" + "\n".join(f"- {fact}" for fact in facts))
        if chunk_text:
            content_parts.append("## 来源片段\n" + chunk_text)
        if not content_parts:
            continue
        results.append({
            "id": f"kg:{doc_id}",
            "content": "\n\n".join(content_parts),
            "distance": rank,
            "score": min(1.0, doc_scores.get(doc_id, 0.0) / 4.0),
            "source": doc.filename if doc else "knowledge_graph",
            "retrieval_source": "knowledge_graph",
            "document_id": doc_id,
            "kg_score": doc_scores.get(doc_id, 0.0),
            "kg_facts": facts,
        })
    return results


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


def extraction_preview(text: str, filename: str = "") -> dict[str, Any]:
    extraction = extract_spirit_kg(text, filename=filename)
    return {
        "entity": asdict(extraction.entity) if extraction.entity else None,
        "nodes": [asdict(n) for n in extraction.nodes],
        "facts": [asdict(f) for f in extraction.facts],
        "llm_candidates": extraction.llm_candidates,
    }


async def _match_entities_for_query(session: AsyncSession, query: str, limit: int) -> list[KGEntity]:
    terms = _query_terms(query)
    clauses = []
    for term in terms:
        if len(term) < 2:
            continue
        clauses.append(KGEntity.name.ilike(f"%{term}%"))
        clauses.append(KGEntity.normalized_name.ilike(f"%{term.lower()}%"))
    candidates: list[KGEntity] = []
    if clauses:
        r = await session.execute(select(KGEntity).where(or_(*clauses)).limit(limit * 3))
        candidates.extend(r.scalars().all())

    # Chinese queries often embed entity names without separators: "虫刺有哪些精灵会".
    scan_r = await session.execute(
        select(KGEntity).where(KGEntity.type.in_(["spirit", "skill", "attribute", "bloodline_skill", "skill_stone"]))
    )
    seen_ids = {e.id for e in candidates}
    for entity in scan_r.scalars().all():
        if entity.id in seen_ids:
            continue
        if entity.name in _ENTITY_STOPWORDS:
            continue
        if len(entity.name) >= 2 and entity.name in query:
            candidates.append(entity)
            seen_ids.add(entity.id)
    if not candidates:
        return []

    def score(entity: KGEntity) -> tuple[int, int]:
        name = entity.name
        if name in _ENTITY_STOPWORDS:
            return -999, 0
        exact = 3 if name == query else 0
        contains = 2 + min(len(name), 8) if name and name in query else 0
        type_bonus = 1 if entity.type in {"spirit", "skill", "attribute"} else 0
        return exact + contains + type_bonus, len(name)

    return sorted(candidates, key=score, reverse=True)[:limit]


def _query_terms(query: str) -> list[str]:
    terms = re.findall(r"[\w\u4e00-\u9fff]+", query)
    # Chinese entity names often appear embedded in longer phrases; keep the whole query too.
    compact = re.sub(r"[^\w\u4e00-\u9fff]+", "", query)
    if compact and compact not in terms:
        terms.append(compact)
    return list(dict.fromkeys(t.strip() for t in terms if t.strip()))


def _format_fact(subject: KGEntity | None, rel: KGRelation, obj: KGEntity | None) -> str:
    if not subject:
        return ""
    target = obj.name if obj else rel.value_text
    if target is None and rel.value_number is not None:
        target = str(rel.value_number)
    if target is None and (rel.value_min is not None or rel.value_max is not None):
        target = f"{rel.value_min or ''}~{rel.value_max or ''}"
    return f"{subject.name} --{rel.predicate}--> {target or ''}".strip()


def _relation_relevance(
    query: str,
    terms: list[str],
    rank: int,
    subject: KGEntity | None,
    rel: KGRelation,
    obj: KGEntity | None,
) -> float:
    score = 1.0 / (1 + rank * 0.2)
    subject_name = subject.name if subject else ""
    object_name = obj.name if obj else (rel.value_text or "")
    predicate = rel.predicate or ""

    if subject_name and subject_name in query:
        score += 1.2
    if object_name and object_name in query:
        score += 1.4
    if predicate and predicate in query:
        score += 1.1
    for term in terms:
        if len(term) < 2:
            continue
        if subject_name and term in subject_name:
            score += 0.25
        if object_name and term in object_name:
            score += 0.35
        if predicate and term in predicate:
            score += 0.3

    score += _CORE_PREDICATE_BONUS.get(predicate, 0.0)
    if any(marker in query for marker in ("会", "拥有", "掌握", "技能")):
        if predicate == "拥有技能":
            score += 0.6
        elif predicate == "可学技能石":
            score -= 0.25
    if rel.extractor == "rule":
        score += 0.15
    elif rel.extractor == "llm":
        score += 0.05 * float(rel.confidence or 0.0)
    return score


def _entity_dict(entity: KGEntity) -> dict[str, Any]:
    return {
        "id": entity.id,
        "name": entity.name,
        "type": entity.type,
        "properties": entity.properties or {},
        "source_document_id": entity.source_document_id,
    }


def _relation_dict(rel: KGRelation, subject: KGEntity | None, obj: KGEntity | None) -> dict[str, Any]:
    return {
        "id": rel.id,
        "subject": _entity_dict(subject) if subject else None,
        "predicate": rel.predicate,
        "object": _entity_dict(obj) if obj else None,
        "value_text": rel.value_text,
        "value_number": rel.value_number,
        "value_min": rel.value_min,
        "value_max": rel.value_max,
        "confidence": rel.confidence,
        "extractor": rel.extractor,
        "source_document_id": rel.source_document_id,
        "source_chunk_id": rel.source_chunk_id,
    }


async def _find_entity(session: AsyncSession, name: str) -> KGEntity | None:
    normalized = _key(name)
    r = await session.execute(
        select(KGEntity)
        .where(or_(KGEntity.normalized_name == normalized, KGEntity.name == name))
        .order_by(KGEntity.type.asc())
        .limit(1)
    )
    return r.scalar_one_or_none()


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


def _key(value: str) -> str:
    return normalize_name(value).lower()
