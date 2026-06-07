# -*- coding: utf-8 -*-
"""Persistence helpers for the lightweight knowledge graph."""
from __future__ import annotations

from typing import Any

from sqlalchemy import delete, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.kg.extractor import KGFact, KGNode, normalize_name
from app.infrastructure.database.models import KGEntity, KGRelation


async def clear_kg(session: AsyncSession) -> None:
    await session.execute(delete(KGRelation))
    await session.execute(delete(KGEntity))


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


def _key(value: str) -> str:
    return normalize_name(value).lower()
