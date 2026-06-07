# -*- coding: utf-8 -*-
"""RAG evaluation API routes."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from loguru import logger
from pydantic import BaseModel, Field

from app.core.evaluation.service import (
    generate_testset_file,
    list_eval_artifacts,
    load_eval_report,
    quick_retrieval_evaluation,
    run_ragas_evaluation_file,
)

router = APIRouter(tags=["evaluation"])


class GenerateTestSetRequest(BaseModel):
    """测试集自动生成请求。"""

    sample_size: int = Field(default=50, ge=5, le=500, description="采样文档片段数")
    questions_per_chunk: int = Field(default=2, ge=1, le=5, description="每个片段生成的问题数")
    collection: str = Field(default="agent_knowledge", description="Milvus 集合名")
    testset_name: str = Field(default="", description="数据集名称（留空自动生成）")


class RunRagasEvalRequest(BaseModel):
    """RAGAS 完整评估请求。"""

    testset_path: str = Field(default="", description="测试集 JSON 文件路径")
    eval_mode: str = Field(default="full", description="评估模式：full / retrieval_only / generation_only")
    top_k: int = Field(default=5, ge=1, le=20, description="检索返回数")


@router.get("/evaluate")
async def evaluate_rag(q: str = Query(...), top_k: int = 5) -> dict:
    """RAG 快速评估：检索 Recall + MRR。"""
    return await quick_retrieval_evaluation(q, top_k=top_k)


@router.post("/evaluate/testset/generate")
async def generate_testset(req: GenerateTestSetRequest) -> dict:
    """
    从知识库自动生成 RAGAS 评测数据集。

    从 Milvus 随机采样文档片段，用 LLM 为每个片段生成 (问题, 答案, 关键事实)，
    导出为 JSON 文件供人工抽检后使用。
    """
    try:
        return await generate_testset_file(
            sample_size=req.sample_size,
            questions_per_chunk=req.questions_per_chunk,
            collection=req.collection,
            testset_name=req.testset_name,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("测试集生成失败")
        raise HTTPException(status_code=500, detail=f"测试集生成失败: {exc}") from exc


@router.post("/evaluate/ragas")
async def run_ragas_evaluation(req: RunRagasEvalRequest) -> dict:
    """
    运行完整的 RAGAS 评估。

    根据测试集文件执行检索→生成→评分全流程，输出评估报告。
    """
    try:
        return await run_ragas_evaluation_file(
            testset_path=req.testset_path,
            eval_mode=req.eval_mode,
            top_k=req.top_k,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("RAGAS 评估执行失败")
        raise HTTPException(status_code=500, detail=f"评估执行失败: {exc}") from exc


@router.get("/evaluate/history")
async def list_eval_reports() -> dict:
    """列出所有历史评估报告。"""
    return list_eval_artifacts()


@router.get("/evaluate/report/{filename}")
async def get_eval_report(filename: str) -> dict:
    """获取指定评估报告的详细内容。"""
    try:
        return load_eval_report(filename)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"报告加载失败: {exc}"
        ) from exc
