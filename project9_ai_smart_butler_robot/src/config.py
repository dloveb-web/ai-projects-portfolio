"""
Configuration and constants for the AI Smart Butler Robot.
"""
from typing import Optional, List
from pydantic_settings import BaseSettings, SettingsConfigDict
from enum import Enum


class PersonalityType(str, Enum):
    """Personality types for the robot."""
    ELDERLY = "elderly"
    EFFICIENT = "efficient"
    FUN = "fun"


class EmotionType(str, Enum):
    """Emotion types for recognition."""
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    ANXIOUS = "anxious"
    TIRED = "tired"
    SCARED = "scared"
    CONFUSED = "confused"
    NEUTRAL = "neutral"


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
    
    # Data Storage
    DATA_PATH: str = "data"
    DATABASE_PATH: str = "data/smart_butler.db"
    KNOWLEDGE_BASE_PATH: str = "data/knowledge_base"
    REMINDER_STORAGE_PATH: str = "data/reminders"
    HEALTH_DATA_PATH: str = "data/health_data"
    
    # Local LLM Settings
    USE_LOCAL_LLM: bool = True
    LLM_MODEL_NAME: str = "Qwen/Qwen2-7B-Instruct"
    LLM_MODEL_PATH: str = "models/"
    LLM_CONTEXT_LENGTH: int = 4096
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 1024
    LLM_QUANTIZATION: str = "q4_0"
    
    # Speech Recognition Settings
    WAKE_WORD: str = "你好管家"
    ASR_MODEL_NAME: str = "openai/whisper-small"
    ASR_LANGUAGE: str = "zh"
    ASR_ENABLE_LOCAL: bool = True
    
    # Text-to-Speech Settings
    TTS_MODEL_NAME: str = ""
    TTS_SPEED: float = 1.0
    TTS_VOLUME: float = 1.0
    
    # User Recognition Settings
    FACE_RECOGNITION_ENABLED: bool = True
    VOICEPRINT_RECOGNITION_ENABLED: bool = True
    FACE_RECOGNITION_THRESHOLD: float = 0.6
    VOICEPRINT_THRESHOLD: float = 0.7
    MAX_FAMILY_MEMBERS: int = 6
    
    # Personality Settings
    DEFAULT_PERSONALITY: PersonalityType = PersonalityType.ELDERLY
    PERSONALITY_ELDERLY_SPEED: float = 0.8
    PERSONALITY_EFFICIENT_SPEED: float = 1.0
    PERSONALITY_FUN_SPEED: float = 1.2
    PERSONALITY_SWITCH_DELAY_MS: int = 500
    
    # Emotion Recognition Settings
    EMOTION_RECOGNITION_ENABLED: bool = False
    EMOTION_MODEL_NAME: str = ""
    EMOTION_RESPONSE_ENABLED: bool = True
    
    # Health Monitoring Settings
    HEALTH_MONITORING_ENABLED: bool = True
    TEMPERATURE_SENSOR_ENABLED: bool = True
    BREATHING_MONITORING_ENABLED: bool = True
    HEALTH_DATA_RETENTION_DAYS: int = 90
    TEMPERATURE_ERROR_THRESHOLD: float = 0.3
    BREATHING_ERROR_THRESHOLD: int = 2
    
    # Home Control Settings
    HOME_CONTROL_ENABLED: bool = True
    INFRARED_ENABLED: bool = True
    WIFI_SMART_ENABLED: bool = True
    MATTER_PROTOCOL_ENABLED: bool = False
    HOME_CONTROL_SUCCESS_RATE_TARGET: float = 0.98
    
    # Privacy Protection Settings
    LOCAL_PROCESSING_RATE_TARGET: float = 0.95
    ENCRYPTION_ENABLED: bool = True
    PRIVACY_INDICATOR_ENABLED: bool = True
    DATA_AUTO_CLEANUP: bool = False
    
    # Emergency Contacts
    EMERGENCY_CONTACT_1_NAME: Optional[str] = None
    EMERGENCY_CONTACT_1_PHONE: Optional[str] = None
    EMERGENCY_CONTACT_2_NAME: Optional[str] = None
    EMERGENCY_CONTACT_2_PHONE: Optional[str] = None
    
    # Robot Hardware
    ROBOT_MOBILE: bool = True
    OBSTACLE_AVOIDANCE_ENABLED: bool = True
    AUTO_DOCKING_ENABLED: bool = True
    BATTERY_CAPACITY: int = 5000
    OBSTACLE_AVOIDANCE_ACCURACY_TARGET: float = 0.99
    MIN_OBSTACLE_DISTANCE_CM: int = 10
    
    # Reminder Settings
    REMINDER_FIRST_RETRY_MINUTES: int = 5
    REMINDER_SECOND_RETRY_MINUTES: int = 15
    REMINDER_ESCALATE_MINUTES: int = 30
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_PATH: str = "logs/"
    
    @property
    def API_URL(self) -> str:
        """Dynamically generate API URL from host and port."""
        return f"http://{self.API_HOST}:{self.API_PORT}"


# Global settings instance
settings = Settings()


class PersonalityConfig:
    """Personality configuration constants."""
    
    PERSONALITIES = {
        PersonalityType.ELDERLY: {
            "name": "长辈陪伴型",
            "speed": settings.PERSONALITY_ELDERLY_SPEED,
            "volume": 1.1,
            "font_size": "large",
            "formality": "casual_simple",
            "response_length": "medium",
            "priority_features": ["health_reminders", "medication_reminders"]
        },
        PersonalityType.EFFICIENT: {
            "name": "高效助手型",
            "speed": settings.PERSONALITY_EFFICIENT_SPEED,
            "volume": 1.0,
            "font_size": "normal",
            "formality": "concise",
            "response_length": "short",
            "priority_features": ["home_control", "schedule_management"]
        },
        PersonalityType.FUN: {
            "name": "趣味伙伴型",
            "speed": settings.PERSONALITY_FUN_SPEED,
            "volume": 1.0,
            "font_size": "normal",
            "formality": "playful",
            "response_length": "medium",
            "priority_features": ["storytelling", "games", "learning"]
        }
    }


class EmotionResponseConfig:
    """Emotion response mapping configuration."""
    
    EMOTION_RESPONSES = {
        EmotionType.HAPPY: {
            "response_style": "cheerful",
            "actions": ["share_positive_news", "celebrate"],
            "priority": "low"
        },
        EmotionType.SAD: {
            "response_style": "warm_compassionate",
            "actions": ["offer_listening", "suggest_relaxation"],
            "priority": "medium",
            "notify_trend": True
        },
        EmotionType.ANGRY: {
            "response_style": "calm_neutral",
            "actions": ["give_space", "offer_calm_down_tips"],
            "priority": "medium"
        },
        EmotionType.ANXIOUS: {
            "response_style": "soothing_slow",
            "actions": ["breathing_exercise", "practical_suggestions"],
            "priority": "medium"
        },
        EmotionType.TIRED: {
            "response_style": "gentle_supportive",
            "actions": ["dim_lights", "play_white_noise", "suggest_rest"],
            "priority": "medium"
        },
        EmotionType.SCARED: {
            "response_style": "reassuring_immediate",
            "actions": ["safety_check", "contact_emergency_if_needed"],
            "priority": "high"
        },
        EmotionType.CONFUSED: {
            "response_style": "clear_simple",
            "actions": ["simplify_explanation", "offer_choices"],
            "priority": "low"
        },
        EmotionType.NEUTRAL: {
            "response_style": "natural",
            "actions": [],
            "priority": "none"
        }
    }


class PromptTemplates:
    """Prompt templates for the AI Smart Butler."""
    
    BASE_SYSTEM_PROMPT: str = """你是一个智能家庭管家机器人，友好、贴心、乐于助人。
你的核心目标是帮助家庭成员，特别是老年人，让生活更便捷、更安全。

核心原则：
1. 保护隐私 - 所有敏感数据只在本地处理
2. 主动关怀 - 但不打扰
3. 简洁清晰 - 回答要简单易懂
4. 安全第一 - 异常情况及时通知家人"""

    ELDERLY_PERSONALITY_PROMPT: str = """你是一个贴心的长辈陪伴型机器人。
特点：
- 语速稍慢，语气温暖
- 用词简单、口语化
- 多关心健康和生活
- 多确认，确保理解正确
- 像家人一样亲切"""

    EFFICIENT_PERSONALITY_PROMPT: str = """你是一个高效的助手型机器人。
特点：
- 语速正常，简洁直接
- 信息密度高
- 高效完成任务
- 少废话，多行动"""

    FUN_PERSONALITY_PROMPT: str = """你是一个有趣的伙伴型机器人。
特点：
- 语速稍快，活泼有趣
- 多用生动的表达方式
- 可以讲故事、玩游戏
- 激发好奇心和学习兴趣"""

    MEDICATION_REMINDER_PROMPT: str = """现在是吃药时间。请温和地提醒用户吃药，
并确认是否已经服用。如果用户说吃了，就记录下来；如果没有，
稍后再提醒一次。"""

    EMERGENCY_RESPONSE_PROMPT: str = """检测到可能的紧急情况。请保持冷静，
用清晰、安抚的语气与用户沟通，确认情况，并准备联系紧急联系人。"""

