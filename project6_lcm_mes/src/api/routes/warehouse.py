from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from src.database.session import get_db
from src.database.models import User, BoxRecord, PalletRecord, WarehouseRecord
from src.api.models import BoxRecordBase
from src.api.dependencies import get_current_user

router = APIRouter(prefix="/api/v1/warehouse", tags=["warehouse"])


@router.post("/boxes", status_code=status.HTTP_201_CREATED)
def create_box(
    box: BoxRecordBase,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    db_box = BoxRecord(**box.model_dump(), packed_by=current_user.id)
    db.add(db_box)
    db.commit()
    db.refresh(db_box)
    return db_box


@router.get("/boxes")
def list_boxes(
    skip: int = 0,
    limit: int = 100,
    status: str = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    query = db.query(BoxRecord)
    if status:
        query = query.filter(BoxRecord.status == status)
    boxes = query.offset(skip).limit(limit).all()
    return boxes


@router.post("/pallets", status_code=status.HTTP_201_CREATED)
def create_pallet(
    pallet_code: str,
    box_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    box = db.query(BoxRecord).filter(BoxRecord.id == box_id).first()
    if not box:
        raise HTTPException(status_code=404, detail="Box not found")
    
    db_pallet = PalletRecord(
        pallet_code=pallet_code,
        box_id=box_id,
        loaded_by=current_user.id
    )
    db.add(db_pallet)
    db.commit()
    db.refresh(db_pallet)
    return db_pallet


@router.post("/receive")
def receive_material(
    material_batch_id: int,
    quantity: float,
    location: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    db_record = WarehouseRecord(
        record_type="receive",
        material_batch_id=material_batch_id,
        quantity=quantity,
        location=location,
        operator_id=current_user.id
    )
    db.add(db_record)
    db.commit()
    db.refresh(db_record)
    return db_record
