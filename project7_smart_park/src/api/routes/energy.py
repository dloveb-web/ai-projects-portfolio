#!/usr/bin/env python3
"""
能源管理路由
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from src.database.session import get_db
from src.database.models.energy import EnergyMeter, EnergyConsumption, EnergyStats
from src.api.models import EnergyMeterResponse, EnergyMeterCreate, EnergyConsumptionResponse, EnergyConsumptionCreate

router = APIRouter(prefix="/energy", tags=["能源管理"])


@router.get("/meters", response_model=list[EnergyMeterResponse])
async def list_meters(db: AsyncSession = Depends(get_db)):
    """获取计量表列表"""
    result = await db.execute(select(EnergyMeter))
    return result.scalars().all()


@router.post("/meters", response_model=EnergyMeterResponse, status_code=status.HTTP_201_CREATED)
async def create_meter(meter_in: EnergyMeterCreate, db: AsyncSession = Depends(get_db)):
    """创建计量表"""
    meter = EnergyMeter(
        room_id=meter_in.room_id,
        name=meter_in.name,
        code=meter_in.code,
        meter_type=meter_in.meter_type,
        unit=meter_in.unit,
        installed_at=meter_in.installed_at,
    )
    db.add(meter)
    await db.commit()
    await db.refresh(meter)
    return meter


@router.get("/consumptions", response_model=list[EnergyConsumptionResponse])
async def list_consumptions(meter_id: int = None, db: AsyncSession = Depends(get_db)):
    """获取能耗记录"""
    query = select(EnergyConsumption)
    if meter_id:
        query = query.where(EnergyConsumption.meter_id == meter_id)
    result = await db.execute(query)
    return result.scalars().all()


@router.post("/consumptions", response_model=EnergyConsumptionResponse, status_code=status.HTTP_201_CREATED)
async def create_consumption(consumption_in: EnergyConsumptionCreate, db: AsyncSession = Depends(get_db)):
    """创建能耗记录"""
    consumption = EnergyConsumption(
        meter_id=consumption_in.meter_id,
        value=consumption_in.value,
        unit=consumption_in.unit,
    )
    db.add(consumption)
    await db.commit()
    await db.refresh(consumption)
    return consumption
