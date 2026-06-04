from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class FinetuneTaskCreate(BaseModel):
    model_name: str = Field(..., description="基座模型名称")
    method: str = Field(..., description="微调方法: lora, qlora, adapter, dpo")
    config: Dict[str, Any] = Field(default_factory=dict, description="训练配置")
    train_data_path: Optional[str] = Field(None, description="训练数据路径")


class FinetuneTaskResponse(BaseModel):
    id: int
    model_name: str
    method: str
    status: str
    config: Dict[str, Any]
    metrics: Optional[Dict[str, Any]]
    created_at: datetime
    completed_at: Optional[datetime]


class FinetunedModelResponse(BaseModel):
    id: int
    task_id: int
    model_path: str
    version: str
    performance: Optional[Dict[str, Any]]
    created_at: datetime


class EvaluateRequest(BaseModel):
    model_id: Optional[int] = Field(None, description="模型ID")
    task_type: str = Field(..., description="评测任务类型")
    config: Dict[str, Any] = Field(default_factory=dict)


class EvaluateResponse(BaseModel):
    task_type: str
    metrics: Dict[str, Any]
    created_at: datetime


class AgentCreate(BaseModel):
    agent_type: str = Field(..., description="Agent类型: react, multiagent, visual")
    config: Dict[str, Any] = Field(default_factory=dict)


class AgentChatRequest(BaseModel):
    session_id: int
    user_input: str = Field(..., description="用户输入")
    image: Optional[str] = Field(None, description="图片（base64编码）")


class AgentChatResponse(BaseModel):
    success: bool
    output: Optional[str]
    tools_used: Optional[List[Dict[str, Any]]]
    error: Optional[str] = None


class AgentSessionResponse(BaseModel):
    id: int
    agent_type: str
    config: Dict[str, Any]
    created_at: datetime


class AgentInteractionResponse(BaseModel):
    id: int
    session_id: int
    user_input: str
    agent_output: Optional[str]
    tools_used: Optional[List[Dict[str, Any]]]
    timestamp: datetime

