from pydantic import BaseModel
from typing import Optional, List, Dict, Any


class DefectPredictionRequest(BaseModel):
    product_code: str
    process_step: str
    equipment_id: Optional[int] = None
    operator_id: Optional[int] = None
    environmental_data: Optional[Dict[str, float]] = None


class DefectPredictionResponse(BaseModel):
    product_code: str
    defect_probability: float
    defect_type: Optional[str] = None
    confidence: float
    recommendations: List[str] = []


class EquipmentHealthRequest(BaseModel):
    equipment_id: int
    look_ahead_days: int = 7


class EquipmentHealthResponse(BaseModel):
    equipment_id: int
    equipment_name: str
    health_score: float
    risk_level: str
    predicted_failure_date: Optional[str] = None
    recommended_actions: List[str] = []


class RepairDecisionRequest(BaseModel):
    product_code: str
    defect_type: str
    defect_severity: str
    repair_history: Optional[List[Dict[str, Any]]] = None


class RepairDecisionResponse(BaseModel):
    product_code: str
    should_repair: bool
    repair_cost_estimate: float
    repair_time_estimate: float
    alternative_action: Optional[str] = None
    confidence: float


class YieldPredictionRequest(BaseModel):
    product_type: str
    batch_size: int
    process_parameters: Optional[Dict[str, float]] = None


class YieldPredictionResponse(BaseModel):
    product_type: str
    predicted_yield: float
    confidence: float
    key_factors: List[str] = []
    optimization_suggestions: List[str] = []


class SchedulingOptimizationRequest(BaseModel):
    order_priority: str = "high"
    resource_constraints: Optional[Dict[str, Any]] = None


class SchedulingOptimizationResponse(BaseModel):
    optimized_schedule: List[Dict[str, Any]] = []
    expected_throughput: float
    bottleneck_analysis: List[str] = []
