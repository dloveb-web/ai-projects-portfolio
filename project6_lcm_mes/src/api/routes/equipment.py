from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from src.database.session import get_db
from src.database.models import User, Equipment
from src.api.models import EquipmentBase, EquipmentResponse
from src.api.dependencies import get_current_user

router = APIRouter(prefix="/api/v1/equipment", tags=["equipment"])


@router.get("/list", response_model=List[EquipmentResponse])
def list_equipment(
    skip: int = 0,
    limit: int = 100,
    status: str = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    query = db.query(Equipment)
    if status:
        query = query.filter(Equipment.status == status)
    equipment_list = query.offset(skip).limit(limit).all()
    return equipment_list


@router.post("/list", response_model=EquipmentResponse, status_code=status.HTTP_201_CREATED)
def create_equipment(
    equipment: EquipmentBase,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    db_equipment = Equipment(**equipment.model_dump())
    db.add(db_equipment)
    db.commit()
    db.refresh(db_equipment)
    return db_equipment


@router.get("/{equipment_id}", response_model=EquipmentResponse)
def get_equipment(
    equipment_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    equipment = db.query(Equipment).filter(Equipment.id == equipment_id).first()
    if not equipment:
        raise HTTPException(status_code=404, detail="Equipment not found")
    return equipment


@router.put("/{equipment_id}/status")
def update_equipment_status(
    equipment_id: int,
    status: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    equipment = db.query(Equipment).filter(Equipment.id == equipment_id).first()
    if not equipment:
        raise HTTPException(status_code=404, detail="Equipment not found")
    equipment.status = status
    db.commit()
    db.refresh(equipment)
    return equipment
