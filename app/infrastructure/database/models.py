# -*- coding: utf-8 -*-
"""SQLAlchemy ORM 模型：会话、消息、文档与追踪日志。"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, JSON, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _uuid() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    """声明式基类。"""


class Conversation(Base):
    """会话表：一次用户对话线程。"""

    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=_uuid,
    )
    title: Mapped[str | None] = mapped_column(String(512), nullable=True)
    user_id: Mapped[str | None] = mapped_column(String(128), index=True, nullable=True)
    meta: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
    )


class Message(Base):
    """消息表：会话中的单条消息。"""

    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=_uuid,
    )
    conversation_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        index=True,
    )
    role: Mapped[str] = mapped_column(String(32))
    content: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    meta: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
    )

    conversation: Mapped["Conversation"] = relationship(back_populates="messages")


class Document(Base):
    """文档表：上传的原始文档元数据。"""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=_uuid,
    )
    filename: Mapped[str] = mapped_column(String(1024))
    mime_type: Mapped[str | None] = mapped_column(String(256), nullable=True)
    storage_path: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    status: Mapped[str] = mapped_column(String(64), default="pending")
    meta: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
    )

    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
    )


class DocumentChunk(Base):
    """文档分块表：用于 RAG 的文本块。"""

    __tablename__ = "document_chunks"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=_uuid,
    )
    document_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("documents.id", ondelete="CASCADE"),
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, default=0)
    content: Mapped[str] = mapped_column(Text)
    vector_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    meta: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
    )

    document: Mapped["Document"] = relationship(back_populates="chunks")


class TraceLog(Base):
    """追踪日志表：持久化关键 Span（可与内存 Tracer 配合）。"""

    __tablename__ = "trace_logs"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=_uuid,
    )
    trace_id: Mapped[str] = mapped_column(String(64), index=True)
    span_id: Mapped[str] = mapped_column(String(64), index=True)
    operation: Mapped[str] = mapped_column(String(256))
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
    )


class KGEntity(Base):
    """轻量知识图谱实体表。"""

    __tablename__ = "kg_entities"
    __table_args__ = (
        UniqueConstraint("type", "normalized_name", name="uq_kg_entity_type_name"),
    )

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=_uuid,
    )
    name: Mapped[str] = mapped_column(String(512), index=True)
    normalized_name: Mapped[str] = mapped_column(String(512), index=True)
    type: Mapped[str] = mapped_column(String(64), index=True)
    properties: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    source_document_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("documents.id", ondelete="SET NULL"),
        index=True,
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    outgoing_relations: Mapped[list["KGRelation"]] = relationship(
        foreign_keys="KGRelation.subject_id",
        back_populates="subject",
        cascade="all, delete-orphan",
    )
    incoming_relations: Mapped[list["KGRelation"]] = relationship(
        foreign_keys="KGRelation.object_id",
        back_populates="object",
    )


class KGRelation(Base):
    """轻量知识图谱关系表：主体 - 谓词 - 客体/值。"""

    __tablename__ = "kg_relations"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=_uuid,
    )
    subject_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("kg_entities.id", ondelete="CASCADE"),
        index=True,
    )
    predicate: Mapped[str] = mapped_column(String(128), index=True)
    object_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("kg_entities.id", ondelete="CASCADE"),
        index=True,
        nullable=True,
    )
    value_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    value_number: Mapped[float | None] = mapped_column(Float, nullable=True)
    value_min: Mapped[float | None] = mapped_column(Float, nullable=True)
    value_max: Mapped[float | None] = mapped_column(Float, nullable=True)
    properties: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    source_document_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("documents.id", ondelete="SET NULL"),
        index=True,
        nullable=True,
    )
    source_chunk_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("document_chunks.id", ondelete="SET NULL"),
        index=True,
        nullable=True,
    )
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    extractor: Mapped[str] = mapped_column(String(64), default="rule")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
    )

    subject: Mapped[KGEntity] = relationship(
        foreign_keys=[subject_id],
        back_populates="outgoing_relations",
    )
    object: Mapped[KGEntity | None] = relationship(
        foreign_keys=[object_id],
        back_populates="incoming_relations",
    )
