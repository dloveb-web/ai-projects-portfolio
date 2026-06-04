from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime
from src.database.session import get_db
from src.database.models import User, ProductionOrder, WorkOrder, ProductionRecord, ProductTrace
from src.api.models import (
    ProductionOrderBase, ProductionOrderResponse,
    WorkOrderCreate, WorkOrderResponse,
    ProductTraceResponse
)
from src.api.dependencies import get_current_user

router = APIRouter(prefix="/api/v1/production", tags=["production"])


@router.get("/orders", response_model=List[ProductionOrderResponse])
def list_orders(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    orders = db.query(ProductionOrder).offset(skip).limit(limit).all()
    return orders


@router.post("/orders", response_model=ProductionOrderResponse, status_code=status.HTTP_201_CREATED)
def create_order(
    order: ProductionOrderBase,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    db_order = ProductionOrder(**order.model_dump(), status="pending")
    db.add(db_order)
    db.commit()
    db.refresh(db_order)
    return db_order


@router.get("/workorders", response_model=List[WorkOrderResponse])
def list_workorders(
    skip: int = 0,
    limit: int = 100,
    status: str = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    query = db.query(WorkOrder)
    if status:
        query = query.filter(WorkOrder.status == status)
    workorders = query.offset(skip).limit(limit).all()
    return workorders


@router.post("/workorders", response_model=WorkOrderResponse, status_code=status.HTTP_201_CREATED)
def create_workorder(
    workorder: WorkOrderCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    db_workorder = WorkOrder(**workorder.model_dump())
    db.add(db_workorder)
    db.commit()
    db.refresh(db_workorder)
    return db_workorder


@router.post("/workorders/{wo_id}/start")
def start_workorder(
    wo_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    workorder = db.query(WorkOrder).filter(WorkOrder.id == wo_id).first()
    if not workorder:
        raise HTTPException(status_code=404, detail="Work order not found")
    workorder.status = "running"
    workorder.start_time = datetime.utcnow()
    db.commit()
    db.refresh(workorder)
    return workorder


@router.post("/workorders/{wo_id}/complete")
def complete_workorder(
    wo_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    workorder = db.query(WorkOrder).filter(WorkOrder.id == wo_id).first()
    if not workorder:
        raise HTTPException(status_code=404, detail="Work order not found")
    workorder.status = "completed"
    workorder.completed_quantity = workorder.quantity
    workorder.end_time = datetime.utcnow()
    db.commit()
    db.refresh(workorder)
    return workorder


@router.get("/trace/{code}", response_model=ProductTraceResponse)
def get_product_trace(
    code: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    trace = None
    if len(code) == 8:
        trace = db.query(ProductTrace).filter(ProductTrace.code_8bit == code).first()
    elif len(code) == 31:
        trace = db.query(ProductTrace).filter(ProductTrace.code_31bit == code).first()
    elif len(code) == 71:
        trace = db.query(ProductTrace).filter(ProductTrace.code_71bit == code).first()
    
    if not trace:
        raise HTTPException(status_code=404, detail="Product not found")
    
    return trace
