from fastapi import APIRouter, Depends
from src.ai.models import (
    DefectPredictionRequest,
    DefectPredictionResponse,
    EquipmentHealthRequest,
    EquipmentHealthResponse,
    RepairDecisionRequest,
    RepairDecisionResponse,
    YieldPredictionRequest,
    YieldPredictionResponse,
    SchedulingOptimizationRequest,
    SchedulingOptimizationResponse
)
from src.ai.service import AIService
from src.api.dependencies import get_current_user

router = APIRouter(prefix="/api/v1/ai", tags=["AI"])


@router.post("/defect-prediction", response_model=DefectPredictionResponse)
def predict_defect(
    request: DefectPredictionRequest,
    current_user: dict = Depends(get_current_user)
):
    return AIService.predict_defect(
        product_code=request.product_code,
        process_step=request.process_step,
        equipment_id=request.equipment_id,
        operator_id=request.operator_id,
        environmental_data=request.environmental_data
    )


@router.post("/equipment-health", response_model=EquipmentHealthResponse)
def get_equipment_health(
    request: EquipmentHealthRequest,
    current_user: dict = Depends(get_current_user)
):
    return AIService.predict_equipment_health(
        equipment_id=request.equipment_id,
        look_ahead_days=request.look_ahead_days
    )


@router.post("/repair-decision", response_model=RepairDecisionResponse)
def make_repair_decision(
    request: RepairDecisionRequest,
    current_user: dict = Depends(get_current_user)
):
    return AIService.make_repair_decision(
        product_code=request.product_code,
        defect_type=request.defect_type,
        defect_severity=request.defect_severity,
        repair_history=request.repair_history
    )


@router.post("/yield-prediction", response_model=YieldPredictionResponse)
def predict_yield(
    request: YieldPredictionRequest,
    current_user: dict = Depends(get_current_user)
):
    return AIService.predict_yield(
        product_type=request.product_type,
        batch_size=request.batch_size,
        process_parameters=request.process_parameters
    )


@router.post("/scheduling-optimization", response_model=SchedulingOptimizationResponse)
def optimize_scheduling(
    request: SchedulingOptimizationRequest,
    current_user: dict = Depends(get_current_user)
):
    return AIService.optimize_scheduling(
        order_priority=request.order_priority,
        resource_constraints=request.resource_constraints
    )


@router.get("/defect-trends")
def get_defect_trends(
    hours: int = 24,
    current_user: dict = Depends(get_current_user)
):
    return AIService.analyze_defect_trends(hours=hours)


@router.get("/quality-report")
def get_quality_report(
    product_type: str = "all",
    current_user: dict = Depends(get_current_user)
):
    return AIService.generate_quality_report(product_type=product_type)


@router.get("/dashboard")
def get_ai_dashboard(
    current_user: dict = Depends(get_current_user)
):
    return {
        "defect_trends": AIService.analyze_defect_trends(24),
        "quality_report": AIService.generate_quality_report(),
        "yield_forecast": {
            "today": AIService.predict_yield("LCM", 1000),
            "week": {"predicted_yield": round(0.945, 3), "trend": "stable"}
        },
        "equipment_alerts": [
            AIService.predict_equipment_health(1, 7),
            AIService.predict_equipment_health(2, 7),
            AIService.predict_equipment_health(3, 7)
        ]
    }
