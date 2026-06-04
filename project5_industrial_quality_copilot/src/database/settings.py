from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    database_url: str = "sqlite+aiosqlite:///./data/quality_copilot.db"

    secret_key: str = "dev-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    yolo_model_path: str = "./models/yolov11/best.pt"
    yolo_confidence_threshold: float = 0.5
    yolo_iou_threshold: float = 0.45

    qwen_model_path: str = "Qwen/Qwen-VL-Chat"
    qwen_device: str = "cuda"
    qwen_max_length: int = 2048

    qwen_api_key: str = ""
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_model_name: str = "qwen-vl-plus"

    chroma_persist_directory: str = "./data/chroma_db"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    upload_dir: str = "./data/uploads"
    report_dir: str = "./data/reports"

    log_level: str = "INFO"


settings = Settings()


def get_upload_dir() -> Path:
    path = Path(settings.upload_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_report_dir() -> Path:
    path = Path(settings.report_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path
