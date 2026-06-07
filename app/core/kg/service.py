# -*- coding: utf-8 -*-
"""Compatibility facade for lightweight knowledge graph services."""
from __future__ import annotations

from app.core.kg.builder import build_document_kg, extraction_preview, is_spirit_document
from app.core.kg.enrich import enrich_document_with_llm
from app.core.kg.repository import (
    add_relation,
    clear_kg,
    entity_relations,
    graph_search,
    kg_stats,
    search_entities,
    upsert_entity,
)
from app.core.kg.retriever import retrieve_kg_context

__all__ = [
    "add_relation",
    "build_document_kg",
    "clear_kg",
    "enrich_document_with_llm",
    "entity_relations",
    "extraction_preview",
    "graph_search",
    "is_spirit_document",
    "kg_stats",
    "retrieve_kg_context",
    "search_entities",
    "upsert_entity",
]
