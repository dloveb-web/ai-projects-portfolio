from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class DefectInfo(BaseModel):
    bbox: List[float]
    class_name: str
    confidence: float


class DetectionRequest(BaseModel):
    image: str
    threshold: float = 0.5
    stream_mode: bool = False


class DetectionResponse(BaseModel):
    task_id: str
    has_defect: bool
    defects: List[DefectInfo]
    processing_time: float
    image_with_boxes: Optional[str] = None


class AnalysisRequest(BaseModel):
    task_id: str
    defect_info: DefectInfo
    include_history: bool = True


class CaseInfo(BaseModel):
    id: str
    content: str
    metadata: Dict[str, Any]
    distance: float


class AnalysisResponse(BaseModel):
    defect_type: str
    severity: str
    cause_analysis: str
    similar_cases: List[CaseInfo]
    suggestions: List[str]


class ReportRequest(BaseModel):
    task_id: str
    include_images: bool = True
    include_trend: bool = True


class ReportSummary(BaseModel):
    total_defects: int
    critical_count: int
    pass_rate: float
    top_defects: List[str]


class ReportResponse(BaseModel):
    report_id: str
    file_path: str
    summary: ReportSummary


class KnowledgeSearchRequest(BaseModel):
    query: str
    top_k: int = 5
    category: Optional[str] = None


class KnowledgeAddRequest(BaseModel):
    title: str
    content: str
    category: str = "defect_case"
    metadata: Optional[Dict[str, Any]] = None


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: Dict[str, Any]


class DashboardStats(BaseModel):
    today_inspections: int
    today_defects: int
    pass_rate: float
    avg_processing_time: float
    recent_alerts: List[Dict[str, Any]]


class UserInfo(BaseModel):
    id: int
    username: str
    role: str
    created_at: datetime


class UserCreate(BaseModel):
    username: str
    password: str
    role: str = "engineer"
