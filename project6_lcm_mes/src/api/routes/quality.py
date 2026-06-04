from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from src.database.session import get_db
from src.database.models import User, DefectRecord, RepairRecord
from src.api.models import DefectRecordBase, RepairRecordBase
from src.api.dependencies import get_current_user

router = APIRouter(prefix="/api/v1/quality", tags=["quality"])


@router.post("/defects", status_code=status.HTTP_201_CREATED)
def create_defect(
    defect: DefectRecordBase,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    db_defect = DefectRecord(**defect.model_dump(), detected_by=current_user.id)
    db.add(db_defect)
    db.commit()
    db.refresh(db_defect)
    return db_defect


@router.get("/defects", response_model=List[dict])
def list_defects(
    skip: int = 0,
    limit: int = 100,
    product_code: str = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    query = db.query(DefectRecord)
    if product_code:
        query = query.filter(DefectRecord.product_code.like(f"%{product_code}%"))
    defects = query.offset(skip).limit(limit).all()
    return defects


@router.post("/repairs", status_code=status.HTTP_201_CREATED)
def create_repair(
    repair: RepairRecordBase,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    defect = db.query(DefectRecord).filter(DefectRecord.id == repair.defect_id).first()
    if not defect:
        raise HTTPException(status_code=404, detail="Defect not found")
    
    repair_count = db.query(RepairRecord).filter(RepairRecord.defect_id == repair.defect_id).count()
    
    db_repair = RepairRecord(
        **repair.model_dump(),
        repair_count=repair_count + 1,
        repairer_id=current_user.id
    )
    db.add(db_repair)
    db.commit()
    db.refresh(db_repair)
    return db_repair


@router.get("/repairs/{defect_id}", response_model=List[dict])
def get_defect_repairs(
    defect_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    repairs = db.query(RepairRecord).filter(RepairRecord.defect_id == defect_id).all()
    return repairs
