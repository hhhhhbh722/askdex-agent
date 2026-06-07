# -*- coding: utf-8 -*-
"""KG-backed retrieval context generation."""
from __future__ import annotations

import re
from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

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


async def retrieve_kg_context(session: AsyncSession, query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """从 KG 召回候选文档/事实，不依赖意图识别。"""
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
