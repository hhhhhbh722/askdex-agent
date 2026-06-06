# -*- coding: utf-8 -*-
"""FastAPI 入口：PostgreSQL + Milvus + Redis + Embedding 全链路。"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.api.routes import chat, conversation, document, health, kg
from app.config import get_settings
from app.core.wiring import wire_app
from app.infrastructure.database.models import Base
from app.infrastructure.database.session import configure_session, init_engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    logger.info("🚀 {} ({})", s.app_name, s.app_env)

    engine = init_engine(s.database_url)
    configure_session(engine)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    app.state.engine = engine
    logger.info("✅ PostgreSQL 就绪")

    await wire_app(app)
    yield
    await engine.dispose()
    logger.info("👋 关闭")


def create_app() -> FastAPI:
    s = get_settings()
    app = FastAPI(title=s.app_name, debug=s.debug, lifespan=lifespan)
    app.include_router(health.router, prefix=s.api_prefix)
    app.include_router(chat.router, prefix=s.api_prefix)
    app.include_router(conversation.router, prefix=s.api_prefix)
    app.include_router(document.router, prefix=s.api_prefix)
    app.include_router(kg.router, prefix=s.api_prefix)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                       allow_methods=["*"], allow_headers=["*"])
    return app


app = create_app()
