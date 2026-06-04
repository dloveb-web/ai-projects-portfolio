from pydantic import BaseModel, ConfigDict
from typing import List, Optional
from datetime import datetime


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    id: str
    title: str
    filename: str
    file_size: int
    content_type: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class QueryRequest(BaseModel):
    query: str
    use_hyde: Optional[bool] = True


class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    contexts: List[dict]


class VersionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    id: str
    version_number: int
    created_at: datetime
    description: str


class CompareResponse(BaseModel):
    version1: dict
    version2: dict
    same_content: bool
