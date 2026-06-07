# -*- coding: utf-8 -*-
"""
RAGAS 风格评估指标。

实现 RAG 系统最核心的 6 大评估指标，分为两组：

**检索阶段指标**（评估检索管线质量）::

    RecallAtK / MRR           — 传统命中率指标，不需要 LLM
    context_precision()       — 检索到的文档是否与问题相关（位置加权）
    context_recall()          — 是否检索到了所有必要信息（需要 Ground Truth 句子）

**生成阶段指标**（评估答案质量）::

    faithfulness()            — 答案是否严格基于检索上下文（无幻觉）
    answer_relevancy()        — 答案是否切题（反向生成问题 + 语义相似度对比）
    answer_correctness()      — 答案是否事实正确（需要 Ground Truth 答案）

所有 LLM 依赖型指标均通过 ``LLMJudge`` 调用，支持无 LLM 时的优雅降级。

典型用法::

    from app.core.evaluation.llm_judge import LLMJudge
    from app.core.evaluation.ragas_metrics import (
        context_precision, context_recall,
        faithfulness, answer_relevancy, answer_correctness,
    )

    judge = LLMJudge(llm=my_llm)
    cp = await context_precision(judge, "什么是RAG？", ["RAG是...", "今天天气..."])
    faith = await faithfulness(judge, "RAG是检索增强生成。", ["RAG是检索增强生成技术。"])
"""

from __future__ import annotations

from collections import defaultdict

from loguru import logger

from .llm_judge import LLMJudge


# ======================================================================
# 一、检索阶段指标
# ======================================================================


# ------------------------------------------------------------------
# 1.1 传统检索指标（无需 LLM）
# ------------------------------------------------------------------


def recall_at_k(
    relevant_ids: list[str],
    retrieved_ids: list[str],
    k: int = 3,
) -> float:
    """
    计算 Recall@K：前 K 个检索结果中命中了多少相关文档。

    这是最基础的检索质量指标，不需要 LLM 参与。

    :param relevant_ids: 标注的相关文档 ID 列表
    :param retrieved_ids: 系统检索到的文档 ID 列表（按排名排序）
    :param k: 截断值
    :returns: 命中率 0-1
    """
    if not relevant_ids:
        return 0.0
    rel_set = set(relevant_ids)
    top_k = retrieved_ids[:k]
    hits = sum(1 for rid in top_k if rid in rel_set)
    return hits / min(k, len(rel_set)) if rel_set else 0.0


def mrr(
    relevant_ids: list[str],
    retrieved_ids: list[str],
) -> float:
    """
    计算 MRR（Mean Reciprocal Rank）：第一个相关文档排名的倒数。

    :param relevant_ids: 标注的相关文档 ID 列表
    :param retrieved_ids: 系统检索到的文档 ID 列表（按排名排序）
    :returns: MRR 值 0-1，0 表示未命中任何相关文档
    """
    if not relevant_ids or not retrieved_ids:
        return 0.0
    rel_set = set(relevant_ids)
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in rel_set:
            return 1.0 / rank
    return 0.0


def evaluate_retrieval_batch(
    test_cases: list[dict],
) -> dict[str, float]:
    """
    批量计算检索指标（Recall@1/3/5 + MRR）。

    这是现有 ``RAGEvaluator.evaluate_retrieval`` 的独立函数版本，
    方便在 RAGAS 管道中直接调用。

    :param test_cases: 列表，每项包含 ``relevant_ids`` 和 ``retrieved_ids``
    :returns: 字典，键为指标名，值为均值
    """
    if not test_cases:
        return {"recall_at_1": 0.0, "recall_at_3": 0.0, "recall_at_5": 0.0, "mrr": 0.0}

    total = len(test_cases)
    r1, r3, r5 = 0.0, 0.0, 0.0
    mrr_total = 0.0

    for tc in test_cases:
        rel = tc.get("relevant_ids", [])
        ret = tc.get("retrieved_ids", [])
        r1 += recall_at_k(rel, ret, k=1)
        r3 += recall_at_k(rel, ret, k=3)
        r5 += recall_at_k(rel, ret, k=5)
        mrr_total += mrr(rel, ret)

    return {
        "recall_at_1": r1 / total,
        "recall_at_3": r3 / total,
        "recall_at_5": r5 / total,
        "mrr": mrr_total / total,
    }


# ------------------------------------------------------------------
# 1.2 Context Precision — 上下文精度
# ------------------------------------------------------------------


async def context_precision(
    judge: LLMJudge,
    query: str,
    contexts: list[str],
) -> float:
    """
    上下文精度：检索到的文档是否与问题相关，且位置越靠前权重越高。

    **算法**（RAGAS 标准实现）：

    1. 对每个 context 用 LLM 判定是否与 query 相关（relevant / irrelevant）
    2. 计算 Precision@k（k=1,2,...,N）：前 k 个中相关占比
    3. 加权平均：Precision@k 的权重 = (是否相关) / k
       —— 相关文档出现在后面会得到惩罚

    :param judge: LLM 评判器
    :param query: 用户查询
    :param contexts: 检索到的文档片段列表
    :returns: Context Precision 0-1，若无 contexts 返回 0
    """
    if not contexts:
        return 0.0

    # 逐个判定相关度
    verdicts: list[bool] = []
    for i, ctx in enumerate(contexts):
        is_rel = await _judge_relevance(judge, query, ctx, context_index=i + 1)
        verdicts.append(is_rel)

    # 计算加权 precision
    numerator = 0.0
    denominator = 0.0
    for k in range(1, len(verdicts) + 1):
        # Precision@k
        relevant_in_top_k = sum(1 for v in verdicts[:k] if v)
        pk = relevant_in_top_k / k
        # 权重 = (第 k 个是否相关) / k
        weight = (1.0 if verdicts[k - 1] else 0.0) / k
        numerator += pk * weight
        denominator += weight

    return numerator / denominator if denominator > 0 else 0.0


# ------------------------------------------------------------------
# 1.3 Context Recall — 上下文召回
# ------------------------------------------------------------------


async def context_recall(
    judge: LLMJudge,
    query: str,
    contexts: list[str],
    ground_truth_sentences: list[str],
) -> float:
    """
    上下文召回：Ground Truth 中的关键信息是否被检索到的上下文覆盖。

    **算法**：

    1. 对每个 ground_truth_sentence，用 LLM 判定是否被 contexts 中的任一文档覆盖
    2. Recall = 被覆盖的句子数 / 总句子数

    :param judge: LLM 评判器
    :param query: 用户查询（用于提供背景）
    :param contexts: 检索到的文档片段列表
    :param ground_truth_sentences: 标准答案中的关键句子列表
    :returns: Context Recall 0-1，无 ground_truth 时返回 0
    """
    if not ground_truth_sentences or not contexts:
        return 0.0

    covered = 0
    for gts in ground_truth_sentences:
        is_covered = await _judge_sentence_covered(judge, query, gts, contexts)
        if is_covered:
            covered += 1

    return covered / len(ground_truth_sentences)


# ======================================================================
# 二、生成阶段指标
# ======================================================================


# ------------------------------------------------------------------
# 2.1 Faithfulness — 忠实度
# ------------------------------------------------------------------


async def faithfulness(
    judge: LLMJudge,
    answer: str,
    contexts: list[str],
) -> float:
    """
    答案忠实度：回答是否严格基于检索上下文，没有编造/幻觉。

    **算法**（RAGAS claims-based 方法）：

    1. 将 answer 拆分为原子陈述（claims）
    2. 对每个 claim，用 LLM 判定是否被 contexts 中的信息支撑
    3. Faithfulness = 被支撑的 claims 数 / 总 claims 数

    这是 RAG 系统最重要的指标之一：高忠实度 + 低答案质量 = 检索不足；
    低忠实度 = 模型在编造。

    :param judge: LLM 评判器
    :param answer: RAG 生成的答案
    :param contexts: 检索到的上下文文档列表
    :returns: Faithfulness 0-1，无 claims 时返回 1.0
    """
    if not answer.strip():
        return 0.0
    if not contexts:
        return 0.0

    # Step 1：拆分原子陈述
    claims = await judge.extract_claims(answer)
    if not claims:
        return 0.0

    # Step 2：逐条判定是否被上下文支撑
    supported = 0
    for claim in claims:
        is_supported = await _judge_claim_supported(judge, claim, contexts)
        if is_supported:
            supported += 1

    return supported / len(claims)


# ------------------------------------------------------------------
# 2.2 Answer Relevancy — 答案相关性
# ------------------------------------------------------------------


async def answer_relevancy(
    judge: LLMJudge,
    query: str,
    answer: str,
    *,
    n_questions: int = 3,
    embedding=None,
) -> float:
    """
    答案相关性：答案是否切题。

    **算法**（RAGAS reverse-question 方法）：

    1. 从 answer 反向生成 n 个问题（"这个答案能回答什么问题？"）
    2. 计算每个生成问题与原始 query 的 embedding 余弦相似度
    3. AnswerRelevancy = 相似度均值

    如果未提供 embedding，降级为 LLM 直接打分模式。

    :param judge: LLM 评判器
    :param query: 原始用户查询
    :param answer: RAG 生成的答案
    :param n_questions: 反向生成的问题数（推荐 3）
    :param embedding: 可选的 EmbeddingAPI，用于计算语义相似度
    :returns: Answer Relevancy 0-1
    """
    if not answer.strip():
        return 0.0

    # Step 1：反向生成问题
    generated_questions = await judge.generate_questions(answer, n=n_questions)
    if not generated_questions:
        # 降级：LLM 直接打分
        return await _judge_relevancy_direct(judge, query, answer)

    # Step 2：计算语义相似度
    if embedding is not None:
        try:
            # 计算 query embedding
            query_vec = (await embedding.aencode([query]))[0]
            gen_vecs = await embedding.aencode(generated_questions)

            # 余弦相似度
            similarities: list[float] = []
            for gv in gen_vecs:
                sim = _cosine_similarity(query_vec, gv)
                similarities.append(max(0.0, sim))

            avg_sim = sum(similarities) / len(similarities)
            return avg_sim

        except Exception as exc:
            logger.warning("answer_relevancy: embedding 计算失败，降级 LLM 打分: {}", exc)

    # 降级：LLM 直接打分
    return await _judge_relevancy_direct(judge, query, answer)


# ------------------------------------------------------------------
# 2.3 Answer Correctness — 答案正确性
# ------------------------------------------------------------------


async def answer_correctness(
    judge: LLMJudge,
    answer: str,
    ground_truth_answer: str,
) -> float:
    """
    答案正确性：生成答案与标准答案的事实一致程度。

    **算法**（RAGAS 方法）：

    1. 用 LLM 对比 answer 和 ground_truth_answer
    2. 从两个维度打分：
       - TP（正确陈述了多少 GT 中的信息）
       - FP（编造了多少 GT 中没有的信息）
    3. F1-style 综合得分

    如果无 Ground Truth，此指标不可计算。

    :param judge: LLM 评判器
    :param answer: RAG 生成的答案
    :param ground_truth_answer: 人工标注/LLM 生成的标准答案
    :returns: Answer Correctness 0-1
    """
    if not answer.strip() or not ground_truth_answer.strip():
        return 0.0

    prompt = (
        "你是严格的事实核查员。请对比「生成答案」和「标准答案」"
        "，从以下两个维度评分：\n\n"
        f"## 生成答案\n{answer.strip()[:2000]}\n\n"
        f"## 标准答案\n{ground_truth_answer.strip()[:2000]}\n\n"
        "## 评分维度\n"
        "1. TP（正确信息覆盖度）：生成答案正确陈述了多少标准答案中的关键信息？0-1 打分\n"
        "2. FP（错误/编造程度）：生成答案中有多少信息是标准答案中没有的？（0=完全未编造, 1=严重编造）\n\n"
        "## 输出格式\n"
        '请严格输出 JSON：{"tp": 0.X, "fp": 0.Y}\n'
        "只输出 JSON，不要有其他内容："
    )

    import json

    for _attempt in range(3):
        try:
            raw = await judge._llm.acomplete(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            # 提取 JSON
            import re
            match = re.search(r"\{.*?\}", raw, re.DOTALL)
            if match:
                scores = json.loads(match.group())
                tp = max(0.0, min(1.0, float(scores.get("tp", 0.5))))
                fp = max(0.0, min(1.0, float(scores.get("fp", 0.5))))
                # F1-style：TP 高 + FP 低 = 高分
                if tp + fp == 0:
                    return 0.0
                f1 = 2 * tp * (1 - fp) / (tp + (1 - fp)) if (tp + (1 - fp)) > 0 else 0.0
                return max(0.0, min(1.0, f1))
        except Exception as exc:
            logger.warning("answer_correctness: LLM 调用失败: {}", exc)

    return 0.5


# ======================================================================
# 三、内部辅助函数
# ======================================================================


async def _judge_relevance(
    judge: LLMJudge,
    query: str,
    context: str,
    context_index: int = 1,
) -> bool:
    """
    判定单个 context 是否与 query 相关。
    """
    prompt = (
        "你是一个相关性判定助手。请判定以下文档片段是否与用户问题「相关」。\n\n"
        f"用户问题：{query[:500]}\n\n"
        f"文档片段[{context_index}]：{context[:1000]}\n\n"
        "如果该片段包含能帮助回答问题的信息，则为「相关」；"
        "如果内容不相关或信息不足以回答问题，则为「不相关」。\n"
        "请只回答「相关」或「不相关」两个词："
    )
    result = await judge.classify(prompt, labels=["相关", "不相关"])
    return result == "相关"


async def _judge_sentence_covered(
    judge: LLMJudge,
    query: str,
    ground_truth_sentence: str,
    contexts: list[str],
) -> bool:
    """
    判定 ground_truth_sentence 中的信息是否被 contexts 任一片段覆盖。
    """
    # 合并上下文（截断以防止 prompt 过长）
    combined = "\n".join(
        f"[{i+1}] {ctx[:300]}" for i, ctx in enumerate(contexts[:10])
    )

    prompt = (
        "你是一个信息覆盖度判定助手。请判定「关键信息句」的内容是否被以下「检索上下文」中的任意片段所覆盖。\n\n"
        f"用户原始问题：{query[:300]}\n\n"
        f"关键信息句：{ground_truth_sentence[:500]}\n\n"
        f"检索上下文：\n{combined[:3000]}\n\n"
        "如果关键信息句的核心事实在检索上下文的任意片段中被提及或可推断，"
        "则为「已覆盖」；否则为「未覆盖」。\n"
        "请只回答「已覆盖」或「未覆盖」两个词："
    )
    result = await judge.classify(prompt, labels=["已覆盖", "未覆盖"])
    return result == "已覆盖"


async def _judge_claim_supported(
    judge: LLMJudge,
    claim: str,
    contexts: list[str],
) -> bool:
    """
    判定单个原子陈述（claim）是否被上下文信息支撑。
    """
    combined = "\n".join(
        f"[{i+1}] {ctx[:300]}" for i, ctx in enumerate(contexts[:10])
    )

    prompt = (
        "你是一个事实核查助手。请判定以下「陈述」是否能在「参考上下文」中找到依据。\n\n"
        f"陈述：{claim[:800]}\n\n"
        f"参考上下文：\n{combined[:3000]}\n\n"
        "判定标准：\n"
        "- 「有依据」：陈述的核心信息在上下文中被明确提及或可直接推断\n"
        "- 「无依据」：陈述的信息在上下文中无法找到支撑，或明显超出上下文范围\n"
        "请只回答「有依据」或「无依据」两个词："
    )
    result = await judge.classify(prompt, labels=["有依据", "无依据"])
    return result == "有依据"


async def _judge_relevancy_direct(
    judge: LLMJudge,
    query: str,
    answer: str,
) -> float:
    """
    降级方案：LLM 直接打分判断答案与问题的相关度。
    """
    prompt = (
        "你是一个答案质量评估助手。请评估以下「答案」与「用户问题」的相关程度。\n\n"
        f"用户问题：{query[:500]}\n\n"
        f"答案：{answer[:1000]}\n\n"
        "评分标准：\n"
        "- 1.0：答案完全切题，直接回应用户问题\n"
        "- 0.5：答案部分相关，但包含不必要的信息或遗漏关键点\n"
        "- 0.0：答案完全不相关或答非所问\n\n"
        "请只输出一个 0-1 之间的数字（可保留两位小数）："
    )
    return await judge.score(prompt)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    计算两个向量的余弦相似度。

    :param a: 向量 A
    :param b: 向量 B
    :returns: 余弦相似度 [-1, 1]
    """
    if not a or not b or len(a) != len(b):
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)
