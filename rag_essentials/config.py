from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict, YamlConfigSettingsSource


class AppConfig(BaseModel):
    name: str = "rag-essentials"
    pdf_path: str = "sample.pdf"
    collection_name: str = "pdf_rag_collection"
    reset_collection_on_ingest: bool = False


class RetrievalConfig(BaseModel):
    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k: int = 4
    max_context_chunks: int = 4

    @field_validator("chunk_overlap")
    @classmethod
    def validate_chunk_overlap(cls, value: int, info):
        chunk_size = info.data.get("chunk_size", 1000)
        if value >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return value


class ChromaConfig(BaseModel):
    host: str = "localhost"
    port: int = 8000
    ssl: bool = False
    hnsw_space: str = "cosine"
    collection_metadata: dict[str, str] = Field(default_factory=dict)


class LLMConfig(BaseModel):
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    system_prompt: str = (
        "You are a helpful RAG assistant. Answer only from the supplied context. "
        "If the answer is not found in the context, say so clearly."
    )


class LoggingConfig(BaseModel):
    level: str = "INFO"
    console: bool = True
    log_file: str = "logs/rag_essentials.log"


class Settings(BaseSettings):
    app: AppConfig = Field(default_factory=AppConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    chroma: ChromaConfig = Field(default_factory=ChromaConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        yaml_file="config.yaml",
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv(override=False)
    return Settings()


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent
