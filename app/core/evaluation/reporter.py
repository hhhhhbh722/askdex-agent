# -*- coding: utf-8 -*-
"""
RAGAS 评估报告：汇总、导出与可视化。

``EvalReport`` 是评估流程的最终产物，包含：

- **summary**：各指标的汇总统计（均值、中位数、标准差、可用率）
- **per_query**：每条测试用例的详细得分和上下文
- **检索指标**：Recall@1/3/5、MRR（如果计算了）
- **导出**：JSON（结构化）、Markdown（人类可读）、Dict（程序调用）

典型用法::

    report = await runner.run_full(testset)

    # 查看摘要
    print(report.summary["faithfulness"]["mean"])

    # 导出
    report.to_json("report.json")
    report.to_markdown("report.md")

    # 找出最差的用例
    worst = report.worst_cases("faithfulness", n=5)
"""

from __future__ import annotations

import json as json_lib
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from .runner import PerQueryResult


# ======================================================================
# 报告数据类
# ======================================================================


@dataclass
class EvalReport:
    """
    评估报告。

    属性:
        testset_name: 评测数据集名称
        run_at: 运行时间戳
        summary: 各指标汇总统计
            ``{"faithfulness": {"mean": 0.85, "median": 0.88, "std": 0.12, "available": 45}}``
        retrieval_metrics: 检索命中率指标（Recall@K, MRR）
        per_query: 每条用例的详细结果
        metadata: 自定义元数据（模型名称、检索配置等）
    """

    testset_name: str = ""
    run_at: str = ""
    summary: dict[str, dict[str, float]] = field(default_factory=dict)
    retrieval_metrics: dict[str, float] = field(default_factory=dict)
    per_query: list[PerQueryResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # ==================================================================
    # 工厂方法
    # ==================================================================

    @classmethod
    def from_per_query(
        cls,
        testset_name: str = "",
        per_query: list[PerQueryResult] | None = None,
        retrieval_metrics: dict[str, float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "EvalReport":
        """
        从逐条结果列表构建报告，自动计算各指标的汇总统计。

        :param testset_name: 数据集名称
        :param per_query: 逐条评估结果
        :param retrieval_metrics: 检索命中率（Recall@K, MRR）
        :param metadata: 额外元数据
        :returns: 构建好的 EvalReport
        """
        per_query = per_query or []
        retrieval_metrics = retrieval_metrics or {}

        # 从逐条结果中提取所有指标名
        all_metric_names: set[str] = set()
        for pq in per_query:
            all_metric_names.update(pq.metrics.keys())

        # 汇总每个指标
        summary: dict[str, dict[str, float]] = {}
        for metric_name in sorted(all_metric_names):
            values = [pq.metrics[metric_name] for pq in per_query if metric_name in pq.metrics]
            if values:
                summary[metric_name] = _summarize_values(values)

        # 合并检索指标到 summary
        for k, v in retrieval_metrics.items():
            summary[k] = {"mean": v, "median": v, "std": 0.0, "available": len(per_query)}

        return cls(
            testset_name=testset_name,
            run_at=datetime.now().isoformat(),
            summary=summary,
            retrieval_metrics=retrieval_metrics,
            per_query=per_query,
            metadata=metadata or {},
        )

    # ==================================================================
    # 导出：JSON
    # ==================================================================

    def to_dict(self) -> dict[str, Any]:
        """
        转为可 JSON 序列化的字典。

        :returns: 完整报告字典
        """
        return {
            "testset_name": self.testset_name,
            "run_at": self.run_at,
            "summary": self.summary,
            "retrieval_metrics": self.retrieval_metrics,
            "metadata": self.metadata,
            "per_query": [
                {
                    "query": pq.query,
                    "answer": pq.answer,
                    "contexts": pq.contexts,
                    "metrics": pq.metrics,
                    "ground_truth_answer": pq.ground_truth_answer,
                    "relevant_ids": pq.relevant_ids,
                    "retrieved_ids": pq.retrieved_ids,
                    "latency_ms": pq.latency_ms,
                    "error": pq.error,
                }
                for pq in self.per_query
            ],
        }

    def to_json(self, path: str | Path) -> None:
        """
        导出为 JSON 文件（UTF-8，缩进 2，中文不转义）。

        :param path: 输出文件路径
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json_lib.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info("EvalReport JSON 已导出: {}", p)

    @classmethod
    def from_json(cls, path: str | Path) -> "EvalReport":
        """
        从 JSON 文件加载报告。

        :param path: JSON 文件路径
        :returns: EvalReport 实例
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json_lib.load(f)

        return cls(
            testset_name=data.get("testset_name", ""),
            run_at=data.get("run_at", ""),
            summary=data.get("summary", {}),
            retrieval_metrics=data.get("retrieval_metrics", {}),
            metadata=data.get("metadata", {}),
            per_query=[
                PerQueryResult(
                    query=pq.get("query", ""),
                    answer=pq.get("answer", ""),
                    contexts=pq.get("contexts", []),
                    metrics=pq.get("metrics", {}),
                    ground_truth_answer=pq.get("ground_truth_answer", ""),
                    relevant_ids=pq.get("relevant_ids", []),
                    retrieved_ids=pq.get("retrieved_ids", []),
                    latency_ms=pq.get("latency_ms", 0.0),
                    error=pq.get("error", ""),
                )
                for pq in data.get("per_query", [])
            ],
        )

    # ==================================================================
    # 导出：Markdown
    # ==================================================================

    def to_markdown(self, path: str | Path | None = None) -> str:
        """
        生成人类可读的 Markdown 报告。

        :param path: 可选输出路径，不传则仅返回字符串
        :returns: Markdown 文本
        """
        lines: list[str] = []

        # 标题
        lines.append(f"# RAGAS 评估报告：{self.testset_name}")
        lines.append("")
        lines.append(f"**运行时间**：{self.run_at}")
        lines.append(f"**用例总数**：{len(self.per_query)}")
        lines.append("")

        # 总览表
        lines.append("## 📊 指标总览")
        lines.append("")
        lines.append("| 指标 | 均值 | 中位数 | 标准差 | 可用用例 |")
        lines.append("|------|------|--------|--------|----------|")

        for metric_name, stats in self.summary.items():
            mean = stats.get("mean", 0)
            median = stats.get("median", 0)
            std = stats.get("std", 0)
            available = int(stats.get("available", 0))

            # 用 emoji 标记得分等级
            emoji = _score_emoji(mean)
            lines.append(
                f"| {emoji} {metric_name} "
                f"| {mean:.4f} "
                f"| {median:.4f} "
                f"| {std:.4f} "
                f"| {available} |"
            )

        lines.append("")

        # 检索指标（如果独立计算了）
        if self.retrieval_metrics:
            lines.append("## 🔍 检索命中率")
            lines.append("")
            lines.append("| 指标 | 值 |")
            lines.append("|------|----|")
            for k, v in self.retrieval_metrics.items():
                lines.append(f"| {k} | {v:.4f} |")
            lines.append("")

        # 延迟统计
        latencies = [pq.latency_ms for pq in self.per_query if pq.latency_ms > 0]
        if latencies:
            avg_lat = sum(latencies) / len(latencies)
            max_lat = max(latencies)
            min_lat = min(latencies)
            lines.append("## ⏱️ 延迟统计")
            lines.append("")
            lines.append(f"- 平均延迟：{avg_lat:.1f} ms")
            lines.append(f"- 最大延迟：{max_lat:.1f} ms")
            lines.append(f"- 最小延迟：{min_lat:.1f} ms")
            lines.append("")

        # 逐条详情
        lines.append("## 📋 逐条详情")
        lines.append("")

        for i, pq in enumerate(self.per_query, start=1):
            lines.append(f"### {i}. {pq.query[:80]}{'...' if len(pq.query) > 80 else ''}")
            lines.append("")

            if pq.error:
                lines.append(f"⚠️ **错误**：{pq.error}")
                lines.append("")
                continue

            # 指标得分表
            if pq.metrics:
                lines.append("| 指标 | 得分 |")
                lines.append("|------|------|")
                for k, v in pq.metrics.items():
                    emoji = _score_emoji(v)
                    lines.append(f"| {emoji} {k} | {v:.4f} |")
                lines.append("")

            # 答案摘要
            if pq.answer:
                answer_preview = pq.answer[:300]
                lines.append(f"**回答**：{answer_preview}")
                if len(pq.answer) > 300:
                    lines.append("...")
                lines.append("")

            # 上下文摘要
            if pq.contexts:
                lines.append(f"**上下文（共 {len(pq.contexts)} 段）**：")
                for j, ctx in enumerate(pq.contexts[:3], start=1):
                    lines.append(f"- [{j}] {ctx[:150]}...")
                if len(pq.contexts) > 3:
                    lines.append(f"- ... 还有 {len(pq.contexts) - 3} 段")
                lines.append("")

            lines.append(f"⏱️ 延迟：{pq.latency_ms:.0f} ms")
            lines.append("")

        markdown_text = "\n".join(lines)

        # 写入文件
        if path:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                f.write(markdown_text)
            logger.info("EvalReport Markdown 已导出: {}", p)

        return markdown_text

    # ==================================================================
    # 分析方法
    # ==================================================================

    def worst_cases(self, metric: str, n: int = 5) -> list[PerQueryResult]:
        """
        找出某个指标得分最低的 n 条用例。

        :param metric: 指标名，如 "faithfulness"
        :param n: 返回数量
        :returns: 得分最低的 PerQueryResult 列表（升序排列）
        """
        with_metric = [pq for pq in self.per_query if metric in pq.metrics]
        sorted_cases = sorted(with_metric, key=lambda pq: pq.metrics.get(metric, 1.0))
        return sorted_cases[:n]

    def best_cases(self, metric: str, n: int = 5) -> list[PerQueryResult]:
        """
        找出某个指标得分最高的 n 条用例。

        :param metric: 指标名
        :param n: 返回数量
        :returns: 得分最高的 PerQueryResult 列表（降序排列）
        """
        with_metric = [pq for pq in self.per_query if metric in pq.metrics]
        sorted_cases = sorted(
            with_metric,
            key=lambda pq: pq.metrics.get(metric, 0.0),
            reverse=True,
        )
        return sorted_cases[:n]

    def metric_distribution(self, metric: str, bins: int = 5) -> dict[str, int]:
        """
        计算某个指标的分数分布（直方图）。

        :param metric: 指标名
        :param bins: 分桶数
        :returns: 桶范围 → 用例数，如 ``{"0.0-0.2": 3, "0.2-0.4": 5, ...}``
        """
        values = [pq.metrics[metric] for pq in self.per_query if metric in pq.metrics]
        if not values:
            return {}

        distribution: dict[str, int] = {}
        for i in range(bins):
            low = i / bins
            high = (i + 1) / bins
            label = f"{low:.1f}-{high:.1f}"
            count = sum(1 for v in values if low <= v < high)
            distribution[label] = count

        # 确保 1.0 也落在最后一个桶
        distribution[f"{1 - 1/bins:.1f}-1.0"] += sum(1 for v in values if v >= 1.0)

        return distribution

    # ==================================================================
    # 便捷属性
    # ==================================================================

    @property
    def overall_score(self) -> float:
        """
        综合得分：所有指标均值的均值。

        :returns: 0-1 之间的综合分，无指标时返回 0
        """
        means = [s["mean"] for s in self.summary.values() if "mean" in s]
        return sum(means) / len(means) if means else 0.0

    @property
    def error_count(self) -> int:
        """评估过程中出现错误的用例数。"""
        return sum(1 for pq in self.per_query if pq.error)

    def __repr__(self) -> str:
        n = len(self.per_query)
        score = self.overall_score
        metrics = ", ".join(self.summary.keys())
        return f"EvalReport({self.testset_name!r}, {n} cases, overall={score:.3f}, metrics=[{metrics}])"


# ======================================================================
# 内部工具函数
# ======================================================================


def _summarize_values(values: list[float]) -> dict[str, float]:
    """
    计算一组分数的汇总统计。

    :param values: 分数列表
    :returns: ``{"mean": ..., "median": ..., "std": ..., "min": ..., "max": ..., "available": N}``
    """
    n = len(values)
    if n == 0:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "available": 0}

    mean = sum(values) / n
    sorted_vals = sorted(values)
    median = sorted_vals[n // 2] if n % 2 == 1 else (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2

    variance = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(variance)

    return {
        "mean": round(mean, 6),
        "median": round(median, 6),
        "std": round(std, 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
        "available": n,
    }


def _score_emoji(score: float) -> str:
    """
    根据分数返回对应的 emoji 标记。

    :param score: 0-1 之间的分数
    :returns: emoji 字符
    """
    if score >= 0.8:
        return "🟢"
    elif score >= 0.5:
        return "🟡"
    else:
        return "🔴"
