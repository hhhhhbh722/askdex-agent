# -*- coding: utf-8 -*-
"""Application services for RAG evaluation workflows."""
from __future__ import annotations

from pathlib import Path

from loguru import logger

from app.config import get_settings
from app.core.wiring import get_state, rag_search


async def generate_testset_file(
    *,
    sample_size: int,
    questions_per_chunk: int,
    collection: str,
    testset_name: str = "",
) -> dict:
    from app.core.evaluation.testset import TestSetGenerator

    state = get_state()
    milvus = state.get("milvus")
    emb = state.get("embedding")
    llm = state.get("agent_llm")

    if not milvus:
        raise RuntimeError("Milvus 未就绪，无法采样文档")

    generator = TestSetGenerator(llm=llm, milvus=milvus, embedding=emb)
    testset = await generator.generate(
        sample_size=sample_size,
        questions_per_chunk=questions_per_chunk,
        collection=collection,
        testset_name=testset_name,
    )

    settings = get_settings()
    report_dir = Path(settings.eval_report_dir)
    filename = f"testset_{testset.name.replace(' ', '_')}.json"
    output_path = report_dir / filename
    testset.to_json(output_path)

    return {
        "status": "ok",
        "testset_name": testset.name,
        "test_cases_count": len(testset),
        "output_path": str(output_path),
        "sample_queries": [tc.query[:80] for tc in testset.test_cases[:5]],
    }


async def run_ragas_evaluation_file(
    *,
    testset_path: str = "",
    eval_mode: str = "full",
    top_k: int = 5,
) -> dict:
    from app.core.evaluation.llm_judge import LLMJudge
    from app.core.evaluation.runner import RAGEvalRunner
    from app.core.evaluation.testset import EvalTestSet

    settings = get_settings()
    resolved_testset_path = testset_path or str(Path(settings.eval_report_dir) / "eval_testset.json")

    if not Path(resolved_testset_path).exists():
        available = list(Path(settings.eval_report_dir).glob("testset_*.json"))
        raise FileNotFoundError(f"测试集不存在: {resolved_testset_path}。可用测试集: {[p.name for p in available]}")

    testset = EvalTestSet.from_json(resolved_testset_path)
    state = get_state()
    llm = state.get("agent_llm")
    emb = state.get("embedding")

    judge = LLMJudge(
        llm=llm,
        max_retries=settings.eval_llm_max_retries,
    )

    async def _retrieve(query: str) -> list[dict]:
        return await rag_search(query, top_k=top_k)

    async def _generate(query: str, contexts: list[str]) -> str:
        return await _generate_answer(llm, query, contexts)

    runner = RAGEvalRunner(
        judge=judge,
        retrieve_fn=_retrieve if eval_mode in ("full", "retrieval_only") else None,
        generate_fn=_generate if eval_mode in ("full", "generation_only") else None,
        embedding=emb,
        max_concurrency=settings.eval_max_concurrency,
    )

    if eval_mode == "retrieval_only":
        report = await runner.run_retrieval_eval(testset)
    elif eval_mode == "generation_only":
        report = await runner.run_generation_eval(testset)
    else:
        report = await runner.run_full(testset)

    report_dir = Path(settings.eval_report_dir)
    report_file = report_dir / f"ragas_report_{report.run_at.replace(':', '-')}.json"
    report.to_json(report_file)

    return {
        "status": "ok",
        "eval_mode": eval_mode,
        "testset_name": testset.name,
        "total_cases": len(report.per_query),
        "error_count": report.error_count,
        "overall_score": round(report.overall_score, 4),
        "summary": report.summary,
        "report_path": str(report_file),
    }


async def quick_retrieval_evaluation(query: str, top_k: int = 5) -> dict:
    from app.core.evaluation import RAGEvaluator
    from app.core.retrieval.pipeline import retrieval_pipeline

    state = get_state()
    emb, milvus = state.get("embedding"), state.get("milvus")
    if not emb or not milvus:
        return {"error": "Embedding 或 Milvus 未就绪"}

    results = await retrieval_pipeline(
        query=query,
        embedding=emb,
        milvus=milvus,
        collection=state["settings"].milvus_collection_name,
        top_k=top_k,
        hybrid=True,
        enable_hyde=True,
    )

    evaluator = RAGEvaluator()
    eval_result = evaluator.evaluate_retrieval([{
        "query": query,
        "relevant_ids": [r["id"] for r in results[:3]],
        "retrieved_ids": [r["id"] for r in results],
    }])

    return {
        "query": query,
        "results": [
            {
                "id": r["id"],
                "score": r["score"],
                "content": r["content"][:200],
                "source": r.get("source", ""),
            }
            for r in results
        ],
        "metrics": {
            "recall_at_1": round(eval_result.recall_at_1, 4),
            "recall_at_3": round(eval_result.recall_at_3, 4),
            "recall_at_5": round(eval_result.recall_at_5, 4),
            "mrr": round(eval_result.mrr, 4),
        },
    }


def list_eval_artifacts() -> dict:
    settings = get_settings()
    report_dir = Path(settings.eval_report_dir)

    if not report_dir.exists():
        return {"reports": [], "test_sets": []}

    reports = sorted(
        [p.name for p in report_dir.glob("ragas_report_*.json")],
        reverse=True,
    )[:20]
    test_sets = sorted(
        [p.name for p in report_dir.glob("testset_*.json")],
        reverse=True,
    )[:20]

    return {
        "reports": reports,
        "test_sets": test_sets,
        "report_dir": str(report_dir),
    }


def load_eval_report(filename: str) -> dict:
    from app.core.evaluation.reporter import EvalReport

    settings = get_settings()
    report_path = Path(settings.eval_report_dir) / filename

    if not report_path.exists():
        raise FileNotFoundError(f"报告不存在: {filename}")

    report = EvalReport.from_json(report_path)
    return report.to_dict()


async def _generate_answer(llm, query: str, contexts: list[str]) -> str:
    if llm is None:
        return ""
    ctx_block = "\n\n".join(
        f"[{i + 1}] {ctx[:300]}" for i, ctx in enumerate(contexts[:5])
    )
    prompt = (
        "你是严谨的知识助手。仅根据提供的上下文作答。\n\n"
        f"上下文：\n{ctx_block}\n\n"
        f"用户问题：{query}\n\n"
        "请基于上下文回答："
    )
    try:
        resp = await llm.acomplete([{"role": "user", "content": prompt}], temperature=0.0)
        return resp.strip() if isinstance(resp, str) else str(resp)
    except Exception as exc:
        logger.warning("评估答案生成失败: {}", exc)
        return ""
