from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class ChatRequest(BaseModel):
    prompt: str = Field(..., description="用户输入")
    model: str = Field(default="qwen-turbo", description="模型名称")
    system_prompt: Optional[str] = Field(None, description="系统提示词")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class ChatResponse(BaseModel):
    success: bool
    content: Optional[str] = None
    model: str
    latency_ms: float
    cost: Optional[float] = None
    error: Optional[str] = None


class EvaluateRequest(BaseModel):
    model: str = Field(..., description="模型名称")
    benchmark_type: str = Field(..., description="评测类型: general, domain, security")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    save_results: bool = Field(default=True)


class EvaluateResponse(BaseModel):
    model: str
    benchmark_type: str
    total_questions: int
    success_count: int
    success_rate: float
    avg_latency_ms: float
    total_cost: float
    avg_score: float
    category_scores: Dict[str, float]


class CompareModelsRequest(BaseModel):
    models: List[str] = Field(..., description="模型列表")
    benchmark_type: str = Field(default="general", description="评测类型")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class SecurityCheckRequest(BaseModel):
    text: str = Field(..., description="待检测文本")


class SecurityCheckResponse(BaseModel):
    is_safe: bool
    threats: List[Dict[str, Any]]
    threat_level: str
    checked_at: str


class PromptRequest(BaseModel):
    name: str = Field(..., description="提示词名称")
    content: str = Field(..., description="提示词内容")
    category: str = Field(default="general", description="分类")


class ABTestRequest(BaseModel):
    prompt_a: str = Field(..., description="提示词A")
    prompt_b: str = Field(..., description="提示词B")
    test_prompt: str = Field(..., description="测试问题")
    model: str = Field(default="qwen-turbo")
    system_prompt: Optional[str] = None
    blind_mode: bool = Field(default=False, description="盲测模式")


class ReportRequest(BaseModel):
    model: str = Field(..., description="模型名称")
    benchmark_type: str = Field(..., description="评测类型")
    results: Dict[str, Any] = Field(..., description="评测结果")
