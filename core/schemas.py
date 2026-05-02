import uuid
from datetime import datetime
from typing import Dict, Any
from pydantic import BaseModel, Field, HttpUrl

# Namespace for deterministic UUIDs
NAMESPACE_ARTICLES = uuid.UUID('f0000000-0000-0000-0000-000000000000')

def generate_article_id(url: str) -> str:
    return str(uuid.uuid5(NAMESPACE_ARTICLES, url))

def generate_chunk_id(url: str, index: int) -> str:
    return str(uuid.uuid5(NAMESPACE_ARTICLES, f"{url}_{index}"))

class Article(BaseModel):
    url: HttpUrl
    title: str
    markdown_content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    @property
    def article_id(self) -> str:
        return generate_article_id(str(self.url))

class ArticleChunk(BaseModel):
    chunk_id: str
    parent_url: HttpUrl
    text: str
    metadata_headers: Dict[str, str]
    timestamp: datetime
    
    class Config:
        from_attributes = True
