from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import datetime
from src.database.session import get_db
from src.database.models import User, ProductionOrder, WorkOrder, ProductionRecord, Equipment
from src.api.models import DashboardStats
from src.api.dependencies import get_current_user

router = APIRouter(prefix="/api/v1/reports", tags=["reports"])


@router.get("/dashboard", response_model=DashboardStats)
def get_dashboard_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    total_orders = db.query(func.count(ProductionOrder.id)).scalar() or 0
    active_work_orders = db.query(func.count(WorkOrder.id)).filter(
        WorkOrder.status.in_(["running", "pending"])
    ).scalar() or 0
    total_equipment = db.query(func.count(Equipment.id)).scalar() or 0
    running_equipment = db.query(func.count(Equipment.id)).filter(
        Equipment.status == "running"
    ).scalar() or 0
    
    today = datetime.utcnow().date()
    today_start = datetime.combine(today, datetime.min.time())
    today_production = db.query(func.sum(ProductionRecord.output_quantity)).filter(
        ProductionRecord.created_at >= today_start
    ).scalar() or 0
    
    total_output = db.query(func.sum(ProductionRecord.output_quantity)).scalar() or 0
    total_defects = db.query(func.sum(ProductionRecord.defect_quantity)).scalar() or 0
    defect_rate = (total_defects / total_output * 100) if total_output > 0 else 0
    
    return DashboardStats(
        total_orders=total_orders,
        active_work_orders=active_work_orders,
        total_equipment=total_equipment,
        running_equipment=running_equipment,
        today_production=today_production,
        defect_rate=round(defect_rate, 2)
    )
