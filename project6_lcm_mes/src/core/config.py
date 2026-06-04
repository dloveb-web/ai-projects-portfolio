from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    ENV: str = "development"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    DATABASE_URL: str = "sqlite:///./lcm_mes.db"
    REDIS_URL: str = "redis://localhost:6379/0"
    
    SECRET_KEY: str = "lcm-mes-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    MQTT_BROKER: str = "localhost"
    MQTT_PORT: int = 1883
    MQTT_TOPIC: str = "mes/+/data"
    
    MODEL_PATH: str = "models/"
    
    class Config:
        env_file = ".env"


settings = Settings()
