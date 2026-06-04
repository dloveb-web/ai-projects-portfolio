"""
项目配置管理
"""
import os
from dataclasses import dataclass
from typing import List


@dataclass
class Config:
    """全局配置类"""
    
    # 模拟模式开关
    MOCK_MODE: bool = os.getenv("MOCK_MODE", "true").lower() == "true"
    
    # API配置
    API_BASE_URL: str = os.getenv("API_BASE_URL", "http://localhost:8000")
    DASHSCOPE_API_KEY: str = os.getenv("DASHSCOPE_API_KEY", "")
    
    # 数据库配置
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./data/agent_finetuning.db")
    
    # 模型配置
    BASE_MODEL_PATH: str = os.getenv("BASE_MODEL_PATH", "")
    FINETUNED_MODEL_PATH: str = os.getenv("FINETUNED_MODEL_PATH", "./data/finetuned_models")
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "Qwen/Qwen2-0.5B-Instruct")
    
    # CORS配置
    CORS_ORIGINS: List[str] = os.getenv("CORS_ORIGINS", "http://localhost:7860,http://localhost:3000").split(",")
    
    # 日志配置
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


# 全局配置实例
config = Config()


def is_mock_mode() -> bool:
    """检查是否在模拟模式"""
    return config.MOCK_MODE


def get_config() -> Config:
    """获取配置实例"""
    return config
