"""应用配置：基于 pydantic-settings，支持环境变量与 .env 文件。"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = Field(default="askdex-agent", description="服务名称")
    app_env: str = Field(default="development", description="运行环境")
    debug: bool = Field(default=False, description="调试模式")
    api_prefix: str = Field(default="/api/v1", description="API 前缀")
    host: str = Field(default="0.0.0.0", description="监听地址")
    port: int = Field(default=8000, description="监听端口")

    openai_api_key: str = Field(default="", description="OpenAI API Key")
    openai_api_base: str = Field(
        default="https://api.openai.com/v1",
        description="OpenAI 兼容 API Base",
    )
    openai_model: str = Field(default="gpt-4o-mini", description="默认对话模型")

    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/agent_db",
        description="SQLAlchemy 异步数据库 URL（推荐 postgresql+asyncpg）",
    )
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis URL")
    redis_memory_ttl_seconds: int = Field(
        default=24 * 60 * 60,
        description="Redis 短期会话记忆 TTL，秒；<=0 表示不过期",
    )

    milvus_host: str = Field(default="localhost", description="Milvus 主机")
    milvus_port: int = Field(default=19530, description="Milvus 端口")
    milvus_user: str = Field(default="", description="Milvus 用户名")
    milvus_password: str = Field(default="", description="Milvus 密码")
    milvus_collection_name: str = Field(
        default="agent_knowledge",
        description="默认向量集合名",
    )

    # Embedding 模型（可选，默认复用 openai_* 配置）
    embedding_api_key: str = Field(default="", description="Embedding API Key（留空则复用 OPENAI_API_KEY）")
    embedding_api_base: str = Field(default="", description="Embedding API Base（留空则复用 OPENAI_API_BASE）")
    embedding_model: str = Field(default="text-embedding-3-small", description="Embedding 模型名")
    embedding_dim: int = Field(default=1024, description="嵌入向量维度")

    log_level: str = Field(default="INFO", description="日志级别")

    # ---- RAGAS 评估配置 ----
    eval_llm_model: str = Field(
        default="",
        description="评估专用模型名（留空则复用 openai_model）",
    )
    eval_llm_temperature: float = Field(
        default=0.0,
        description="评估 LLM 调用温度（0=确定性输出，保证评分一致性）",
    )
    eval_llm_max_retries: int = Field(
        default=3,
        description="评估 LLM 调用失败最大重试次数",
    )
    eval_max_concurrency: int = Field(
        default=5,
        description="评估时并发 LLM 调用的最大数量（控制频率和成本）",
    )
    eval_report_dir: str = Field(
        default="./eval_reports",
        description="评估报告默认输出目录",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
