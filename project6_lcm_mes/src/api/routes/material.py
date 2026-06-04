from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from src.database.session import get_db
from src.database.models import User, Material, MaterialBatch
from src.api.models import MaterialBatchBase
from src.api.dependencies import get_current_user

router = APIRouter(prefix="/api/v1/material", tags=["material"])


@router.get("/inventory")
def get_inventory(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    materials = db.query(Material).offset(skip).limit(limit).all()
    return materials


@router.post("/batches", status_code=status.HTTP_201_CREATED)
def create_material_batch(
    batch: MaterialBatchBase,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    material = db.query(Material).filter(Material.id == batch.material_id).first()
    if not material:
        raise HTTPException(status_code=404, detail="Material not found")
    
    db_batch = MaterialBatch(**batch.model_dump())
    db.add(db_batch)
    
    material.stock_quantity += batch.quantity
    db.commit()
    db.refresh(db_batch)
    return db_batch


@router.get("/batches")
def list_batches(
    skip: int = 0,
    limit: int = 100,
    material_id: int = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    query = db.query(MaterialBatch)
    if material_id:
        query = query.filter(MaterialBatch.material_id == material_id)
    batches = query.offset(skip).limit(limit).all()
    return batches
