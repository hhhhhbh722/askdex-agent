# -*- coding: utf-8 -*-
"""
RAGAS 评测数据集：Schema 定义 + 自动生成器。

提供两类核心功能：
1. **数据结构**：``EvalTestCase`` / ``EvalTestSet`` 定义评测数据的标准格式，
   支持 JSON 序列化/反序列化，方便存储与人工审查。
2. **自动生成器**：``TestSetGenerator`` 从知识库（Milvus）采样文档片段，
   利用 LLM 自动生成 (问题, 标准答案, 关键事实) 三元组，大幅降低人工标注成本。

典型使用流程::

    # 1. 从知识库生成测试集
    gen = TestSetGenerator(llm=my_llm, milvus=milvus, embedding=emb)
    testset = await gen.generate(
        sample_size=50,
        questions_per_chunk=2,
        collection="agent_knowledge",
    )

    # 2. 导出为 JSON 供人工抽检
    testset.to_json("eval_testset_20240607.json")

    # 3. 从 JSON 加载用于评估
    testset = EvalTestSet.from_json("eval_testset_20240607.json")
"""

from __future__ import annotations

import json as json_lib
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from .llm_judge import JudgeLLMProtocol


# ======================================================================
# 数据结构
# ======================================================================


@dataclass
class EvalTestCase:
    """
    单条 RAGAS 评测用例。

    字段分为两层：

    - **必需字段**：``query`` — 没有查询就无法评估。
    - **可选 Ground Truth 字段**：有了它们才能计算全量指标（Context Recall、
      Answer Correctness 等）。通过 LLM 自动生成时建议至少包含 ``ground_truth_answer``。

    属性:
        query: 用户查询问题（必需）
        ground_truth_contexts: 标准答案上下文 / 关键句式列表，用于 Context Recall 计算
        ground_truth_answer: 标准答案文本，用于 Answer Correctness 计算
        relevant_ids: 预期检索到的文档片段 ID 列表，用于 Recall@K / MRR 计算
        key_facts: 关键事实句列表，用于 Context Recall 的轻量级评估
        metadata: 额外标签（分类、难度、来源文档等），自由扩展
    """

    query: str
    ground_truth_contexts: list[str] = field(default_factory=list)
    ground_truth_answer: str = ""
    relevant_ids: list[str] = field(default_factory=list)
    key_facts: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转为可 JSON 序列化的字典。"""
        return {
            "query": self.query,
            "ground_truth_contexts": self.ground_truth_contexts,
            "ground_truth_answer": self.ground_truth_answer,
            "relevant_ids": self.relevant_ids,
            "key_facts": self.key_facts,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalTestCase":
        """从字典还原。"""
        return cls(
            query=data.get("query", ""),
            ground_truth_contexts=data.get("ground_truth_contexts", []),
            ground_truth_answer=data.get("ground_truth_answer", ""),
            relevant_ids=data.get("relevant_ids", []),
            key_facts=data.get("key_facts", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class EvalTestSet:
    """
    评测数据集：包含多条 ``EvalTestCase`` 及元信息。

    属性:
        name: 数据集名称（如 "2024Q4-知识库评测"）
        test_cases: 评测用例列表
        created_at: 创建时间戳
        version: 数据集版本号，便于追踪迭代
        description: 数据集用途/备注
    """

    name: str = ""
    test_cases: list[EvalTestCase] = field(default_factory=list)
    created_at: str = ""
    version: str = "1.0"
    description: str = ""

    # ---- 序列化 ----

    def to_dict(self) -> dict[str, Any]:
        """转为字典。"""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "created_at": self.created_at or datetime.now().isoformat(),
            "test_cases": [tc.to_dict() for tc in self.test_cases],
        }

    def to_json(self, path: str | Path) -> None:
        """导出为 JSON 文件（UTF-8，缩进 2，中文不转义）。"""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json_lib.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info("EvalTestSet 已导出: {} ({} 条用例)", p, len(self.test_cases))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalTestSet":
        """从字典还原。"""
        return cls(
            name=data.get("name", ""),
            version=data.get("version", "1.0"),
            description=data.get("description", ""),
            created_at=data.get("created_at", ""),
            test_cases=[EvalTestCase.from_dict(tc) for tc in data.get("test_cases", [])],
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "EvalTestSet":
        """从 JSON 文件加载。"""
        with open(path, "r", encoding="utf-8") as f:
            data = json_lib.load(f)
        testset = cls.from_dict(data)
        logger.info("EvalTestSet 已加载: {} ({} 条用例)", path, len(testset.test_cases))
        return testset

    # ---- 便捷方法 ----

    def __len__(self) -> int:
        return len(self.test_cases)

    def __getitem__(self, index: int) -> EvalTestCase:
        return self.test_cases[index]

    def sample(self, n: int) -> "EvalTestSet":
        """随机采样 n 条用例，返回新的子集（用于快速测试）。"""
        sampled = random.sample(self.test_cases, min(n, len(self.test_cases)))
        return EvalTestSet(
            name=f"{self.name} (sampled {n})",
            test_cases=sampled,
            version=self.version,
        )

    def filter_by_metadata(self, key: str, value: Any) -> "EvalTestSet":
        """按 metadata 字段筛选用例。"""
        filtered = [tc for tc in self.test_cases if tc.metadata.get(key) == value]
        return EvalTestSet(
            name=f"{self.name} ({key}={value})",
            test_cases=filtered,
            version=self.version,
        )

    @property
    def has_ground_truth_answer(self) -> bool:
        """检查是否有标准答案（影响 Answer Correctness 是否可计算）。"""
        return any(tc.ground_truth_answer for tc in self.test_cases)

    @property
    def has_relevant_ids(self) -> bool:
        """检查是否有标注的文档 ID（影响 Recall@K 是否可计算）。"""
        return any(tc.relevant_ids for tc in self.test_cases)


# ======================================================================
# 测试集自动生成器
# ======================================================================


class TestSetGenerator:
    """
    从知识库自动生成 RAGAS 评测数据集。

    工作流程::

        1. sample_documents()   → 从 Milvus 随机采样文档片段
        2. generate_qa_pairs()  → 对每个片段用 LLM 生成 (Q, A, key_facts)
        3. build_testset()      → 组装为 EvalTestSet，自动填充 relevant_ids

    使用示例::

        gen = TestSetGenerator(
            llm=my_llm_adapter,
            milvus=milvus_manager,
            embedding=embedding_api,
        )
        testset = await gen.generate(
            sample_size=50,
            questions_per_chunk=2,
            collection="agent_knowledge",
        )
        testset.to_json("eval_testset.json")
    """

    # 默认的 QA 生成 prompt 模板
    DEFAULT_GENERATE_PROMPT = (
        "你是测试数据集生成器。请仔细阅读以下文档内容，"
        "生成{questions_per_chunk}个用户可能提出的问题，并为每个问题写出标准答案。\n\n"
        "## 文档内容\n{content}\n\n"
        "## 要求\n"
        "- 问题应该覆盖文档中的关键信息点\n"
        "- 答案必须严格基于文档内容，不要编造\n"
        "- 问题应尽量具体，避免过于宽泛\n"
        "- 额外提取 {questions_per_chunk} 个「关键事实句」（直接从文档中摘录）\n\n"
        "## 输出格式\n"
        "请严格输出以下 JSON 格式（不要包含其他内容）：\n"
        "{{\n"
        '  "qa_pairs": [\n'
        '    {{\n'
        '      "question": "问题1",\n'
        '      "answer": "基于文档的标准答案1",\n'
        '      "key_facts": ["关键事实句1", "关键事实句2"]\n'
        "    }}\n"
        "  ]\n"
        "}}"
    )

    def __init__(
        self,
        llm: JudgeLLMProtocol | None = None,
        milvus: Any = None,
        embedding: Any = None,
        generate_prompt: str | None = None,
    ) -> None:
        """
        :param llm: LLM 调用接口（用于生成 QA 对）
        :param milvus: MilvusManager 实例（用于采样文档片段）
        :param embedding: EmbeddingAPI 实例（若 Milvus 搜索需要）
        :param generate_prompt: 自定义 QA 生成 prompt 模板，支持 ``{content}`` 和
            ``{questions_per_chunk}`` 占位符
        """
        self._llm = llm
        self._milvus = milvus
        self._embedding = embedding
        self._generate_prompt = generate_prompt or self.DEFAULT_GENERATE_PROMPT

    # ------------------------------------------------------------------
    # 主入口：生成完整测试集
    # ------------------------------------------------------------------

    async def generate(
        self,
        sample_size: int = 50,
        questions_per_chunk: int = 2,
        collection: str = "agent_knowledge",
        testset_name: str = "",
        seed: int | None = 42,
    ) -> EvalTestSet:
        """
        从知识库采样文档片段并生成完整评测数据集。

        :param sample_size: 采样的文档片段数量（最终用例数 = sample_size × questions_per_chunk）
        :param questions_per_chunk: 每个片段生成的问题数
        :param collection: Milvus 集合名称
        :param testset_name: 数据集名称，留空则自动生成
        :param seed: 随机种子（保证可复现）
        :returns: 生成的 EvalTestSet
        """
        if seed is not None:
            random.seed(seed)

        # 1. 采样文档片段
        chunks = await self.sample_documents(
            n=sample_size,
            collection=collection,
        )
        if not chunks:
            logger.warning("TestSetGenerator: 未采样到任何文档片段，返回空测试集")
            return EvalTestSet(name="empty_testset")

        logger.info("TestSetGenerator: 已采样 {} 个片段，开始生成 QA 对...", len(chunks))

        # 2. 逐个片段生成 QA 对
        all_test_cases: list[EvalTestCase] = []
        for i, chunk in enumerate(chunks):
            try:
                cases = await self.generate_qa_pairs(
                    content=chunk.get("content", ""),
                    chunk_id=chunk.get("id", ""),
                    n=questions_per_chunk,
                )
                all_test_cases.extend(cases)
                if (i + 1) % 10 == 0:
                    logger.info("TestSetGenerator: {}/{} 片段已处理", i + 1, len(chunks))
            except Exception as exc:
                logger.warning("TestSetGenerator: 片段 {} QA 生成失败: {}", chunk.get("id", "?"), exc)

        # 3. 组装 EvalTestSet
        name = testset_name or f"auto-generated-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        return EvalTestSet(
            name=name,
            version="1.0",
            description=(
                f"从 {collection} 集合自动生成，"
                f"采样 {len(chunks)} 片段，共 {len(all_test_cases)} 条用例"
            ),
            created_at=datetime.now().isoformat(),
            test_cases=all_test_cases,
        )

    # ------------------------------------------------------------------
    # Step 1：从知识库采样文档片段
    # ------------------------------------------------------------------

    async def sample_documents(
        self,
        n: int = 50,
        collection: str = "agent_knowledge",
        min_content_length: int = 100,
    ) -> list[dict[str, Any]]:
        """
        从 Milvus 中随机采样 n 个足够长的文档片段。

        策略：用一组随机向量检索，收集不重复的结果。
        若 Milvus 不可用，返回空列表。

        :param n: 需要的片段数量
        :param collection: Milvus 集合名
        :param min_content_length: 最小内容长度（过滤过短的片段）
        :returns: 文档片段列表，每个 dict 至少包含 ``id`` 和 ``content``
        """
        if self._milvus is None:
            logger.warning("TestSetGenerator: 未配置 Milvus，无法采样文档")
            return []

        try:
            # 用随机向量做多轮搜索以收集不重复的片段
            seen_ids: set[str] = set()
            chunks: list[dict[str, Any]] = []

            # 每轮取 top_k 个，最多尝试 5 轮
            for _round in range(5):
                if len(chunks) >= n:
                    break

                # 生成随机向量（与 embedding 维度一致）
                dim = getattr(self._embedding, "dim", 1024) if self._embedding else 1024
                import numpy as np
                random_vec = np.random.randn(dim).astype(np.float32).tolist()

                try:
                    results = await self._milvus.search(
                        collection=collection,
                        query_vector=random_vec,
                        top_k=n * 2,
                    )
                except Exception as exc:
                    logger.warning("TestSetGenerator: Milvus 搜索失败: {}", exc)
                    # 尝试 hybrid_search 降级
                    try:
                        results = await self._milvus.hybrid_search(
                            collection=collection,
                            query_vector=random_vec,
                            query_text="",
                            top_k=n * 2,
                        )
                    except Exception:
                        break

                for r in results or []:
                    rid = r.get("id", "")
                    content = r.get("content", "")
                    if rid not in seen_ids and len(content) >= min_content_length:
                        seen_ids.add(rid)
                        chunks.append(r)
                        if len(chunks) >= n:
                            break

            logger.info("TestSetGenerator: 采样完成 {}/{} 个片段", len(chunks), n)
            return chunks

        except Exception as exc:
            logger.warning("TestSetGenerator: 采样失败: {}", exc)
            return []

    # ------------------------------------------------------------------
    # Step 2：为单个片段生成 QA 对
    # ------------------------------------------------------------------

    async def generate_qa_pairs(
        self,
        content: str,
        chunk_id: str = "",
        n: int = 2,
    ) -> list[EvalTestCase]:
        """
        为单个文档片段生成 n 个 (问题, 答案, 关键事实) 三元组。

        :param content: 文档片段文本内容
        :param chunk_id: 片段 ID（自动填入 relevant_ids）
        :param n: 生成的问题数
        :returns: EvalTestCase 列表
        """
        if self._llm is None or not content.strip():
            return []

        prompt = self._generate_prompt.format(
            content=content[:3000],  # 截断过长文本
            questions_per_chunk=n,
        )

        import json as json_lib

        for attempt in range(3):
            try:
                raw = await self._llm.acomplete(
                    [{"role": "user", "content": prompt}],
                    temperature=0.3,
                )

                # 尝试多种解析策略
                data = self._parse_qa_response(raw)
                if data is None:
                    logger.warning("TestSetGenerator: QA 解析失败，raw={}", raw[:200])
                    continue

                qa_pairs = data.get("qa_pairs", [])
                if not qa_pairs:
                    continue

                test_cases: list[EvalTestCase] = []
                for qa in qa_pairs[:n]:
                    question = (qa.get("question") or "").strip()
                    answer = (qa.get("answer") or "").strip()
                    key_facts = qa.get("key_facts", [])

                    if not question:
                        continue

                    test_cases.append(EvalTestCase(
                        query=question,
                        ground_truth_answer=answer,
                        relevant_ids=[chunk_id] if chunk_id else [],
                        key_facts=key_facts if isinstance(key_facts, list) else [],
                        metadata={
                            "source_chunk_id": chunk_id,
                            "generation_method": "llm_auto",
                        },
                    ))

                return test_cases

            except Exception as exc:
                logger.warning("TestSetGenerator: LLM 调用失败 (尝试 {}/3): {}", attempt + 1, exc)

        return []

    # ------------------------------------------------------------------
    # 私有方法
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_qa_response(raw: str) -> dict | None:
        """
        解析 LLM 输出的 QA JSON。

        容忍 LLM 在 JSON 外包裹 markdown 代码块或额外文本。
        """
        import json as json_lib
        import re

        raw = raw.strip()

        # 去除可能的 markdown 代码块标记
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"```\s*$", "", raw)

        # 尝试直接解析
        try:
            return json_lib.loads(raw)  # type: ignore[no-any-return]
        except (json_lib.JSONDecodeError, TypeError):
            pass

        # 用正则提取 JSON 对象
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json_lib.loads(match.group())  # type: ignore[no-any-return]
            except (json_lib.JSONDecodeError, TypeError):
                pass

        return None
