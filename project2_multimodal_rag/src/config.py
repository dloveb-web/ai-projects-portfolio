"""
Configuration and constants for the Multimodal RAG system.
"""
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with validation."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # Document Storage
    DOCUMENT_STORAGE_PATH: str = "data/documents"
    DATABASE_PATH: str = "data/documents.db"
    
    # Embedding Settings
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    USE_VECTOR_SEARCH: bool = True
    HF_HUB_TIMEOUT: int = 300  # seconds
    HF_HUB_DOWNLOAD_TIMEOUT: int = 300
    
    # LayoutLM Settings
    INIT_LAYOUTLM: bool = True
    LAYOUTLM_MODEL_NAME: str = "microsoft/layoutlmv3-base"
    LAYOUTLM_LOAD_TIMEOUT: int = 600  # seconds
    
    # Retriever Settings
    TOP_K_RESULTS: int = 5
    BM25_WEIGHT: float = 0.6
    VECTOR_WEIGHT: float = 0.4
    
    # Chunking Settings
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100
    MIN_CHUNK_SIZE: int = 100
    
    # OCR Settings
    OCR_DEFAULT_LANG: str = "chi_sim+eng"
    OCR_TIMEOUT: int = 5  # seconds
    
    # LLM Settings
    USE_HYDE: bool = True
    TEMPERATURE: float = 0.1
    DASHSCOPE_API_KEY: Optional[str] = None
    
    # File Settings
    SUPPORTED_EXTENSIONS: list[str] = ["pdf", "docx", "doc", "png", "jpg", "jpeg", "txt", "text"]
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50 MB
    
    @property
    def API_URL(self) -> str:
        """Dynamically generate API URL from host and port."""
        return f"http://{self.API_HOST}:{self.API_PORT}"


# Global settings instance
settings = Settings()


class SearchConfig:
    """Search-related configuration constants (alias to settings for backward compatibility)."""
    TOP_K: int = settings.TOP_K_RESULTS
    BM25_WEIGHT: float = settings.BM25_WEIGHT
    VECTOR_WEIGHT: float = settings.VECTOR_WEIGHT
    MIN_RELEVANCE_SCORE: float = 0.0


class ChunkingConfig:
    """Text chunking configuration constants (alias to settings)."""
    CHUNK_SIZE: int = settings.CHUNK_SIZE
    CHUNK_OVERLAP: int = settings.CHUNK_OVERLAP
    MIN_CHUNK_SIZE: int = settings.MIN_CHUNK_SIZE
    SEPARATORS: list[str] = ["\n\n", "\n", ". ", "! ", "? ", "。", "！", "？", " ", ""]


class ModelConfig:
    """Model-related configuration constants (alias to settings)."""
    EMBEDDING_MODEL: str = settings.EMBEDDING_MODEL_NAME
    LAYOUTLM_MODEL: str = settings.LAYOUTLM_MODEL_NAME
    LLM_MODEL: str = "qwen-turbo"
    EMBEDDING_DIMENSION: Optional[int] = None


class FileConfig:
    """File-related configuration constants (alias to settings)."""
    SUPPORTED_EXTENSIONS: list[str] = settings.SUPPORTED_EXTENSIONS
    MAX_FILE_SIZE: int = settings.MAX_FILE_SIZE


class PromptTemplates:
    """Prompt templates for RAG and HyDE."""
    
    RAG_PROMPT: str = """基于以下参考文档内容，回答用户的问题。

参考文档:
{context}

用户问题: {query}

请根据参考文档内容进行回答，并在回答末尾标注引用来源。"""
    
    HYDE_PROMPT: str = """你是一个文档检索专家。请将用户的实际问题改写成一个详细的"假设答案"。

问题：{query}

假设答案（要尽可能详细、使用文档中的专业术语）："""
