from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    # API Keys
    GOOGLE_API_KEY: str
    QDRANT_API_KEY: Optional[str] = None
    
    # Vector DB
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION: str = "articles"
    EMBEDDING_MODEL: str = "models/text-embedding-004"
    
    # Redis / Task Queue
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    # App Settings
    PROJECT_NAME: str = "ArticlesAI"
    
    # Slack Integration
    SLACK_BOT_TOKEN: str
    SLACK_CHANNEL_ID: str
    
    # Security
    API_KEY: str
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()
