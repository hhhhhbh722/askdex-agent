# -*- coding: utf-8 -*-
"""Rule-based KG extraction and rebuild helpers."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.kg.extractor import extract_spirit_kg
from app.core.kg.repository import _key, add_relation, upsert_entity
from app.infrastructure.database.models import Document, DocumentChunk, KGEntity


def is_spirit_document(document: Document) -> bool:
    meta = document.meta or {}
    group = str(meta.get("group") or "")
    parent = str(meta.get("parent_group") or "")
    return parent == "精灵图鉴" or group == "精灵图鉴" or group.startswith("精灵图鉴 /")


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


def extraction_preview(text: str, filename: str = "") -> dict[str, Any]:
    extraction = extract_spirit_kg(text, filename=filename)
    return {
        "entity": asdict(extraction.entity) if extraction.entity else None,
        "nodes": [asdict(n) for n in extraction.nodes],
        "facts": [asdict(f) for f in extraction.facts],
        "llm_candidates": extraction.llm_candidates,
    }
