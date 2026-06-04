#!/usr/bin/env python3
"""
设备管理路由
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from src.database.session import get_db
from src.database.models.device import Device, DeviceType, DeviceData, DeviceAlert
from src.api.models import DeviceResponse, DeviceCreate, DeviceTypeResponse, DeviceTypeCreate

router = APIRouter(prefix="/device", tags=["设备管理"])


@router.get("/types", response_model=list[DeviceTypeResponse])
async def list_device_types(db: AsyncSession = Depends(get_db)):
    """获取设备类型列表"""
    result = await db.execute(select(DeviceType))
    return result.scalars().all()


@router.post("/types", response_model=DeviceTypeResponse, status_code=status.HTTP_201_CREATED)
async def create_device_type(type_in: DeviceTypeCreate, db: AsyncSession = Depends(get_db)):
    """创建设备类型"""
    device_type = DeviceType(
        name=type_in.name,
        category=type_in.category,
        description=type_in.description,
    )
    db.add(device_type)
    await db.commit()
    await db.refresh(device_type)
    return device_type


@router.get("/", response_model=list[DeviceResponse])
async def list_devices(db: AsyncSession = Depends(get_db)):
    """获取设备列表"""
    result = await db.execute(select(Device))
    return result.scalars().all()


@router.get("/{device_id}", response_model=DeviceResponse)
async def get_device(device_id: int, db: AsyncSession = Depends(get_db)):
    """获取单个设备"""
    result = await db.execute(select(Device).where(Device.id == device_id))
    device = result.scalar_one_or_none()
    if not device:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="设备不存在")
    return device


@router.post("/", response_model=DeviceResponse, status_code=status.HTTP_201_CREATED)
async def create_device(device_in: DeviceCreate, db: AsyncSession = Depends(get_db)):
    """创建设备"""
    device = Device(
        device_type_id=device_in.device_type_id,
        room_id=device_in.room_id,
        name=device_in.name,
        code=device_in.code,
        ip_address=device_in.ip_address,
        location=device_in.location,
        metadata=device_in.metadata,
    )
    db.add(device)
    await db.commit()
    await db.refresh(device)
    return device


@router.put("/{device_id}/status")
async def update_device_status(device_id: int, status: str, db: AsyncSession = Depends(get_db)):
    """更新设备状态"""
    result = await db.execute(select(Device).where(Device.id == device_id))
    device = result.scalar_one_or_none()
    if not device:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="设备不存在")
    
    device.status = status
    await db.commit()
    await db.refresh(device)
    return device
