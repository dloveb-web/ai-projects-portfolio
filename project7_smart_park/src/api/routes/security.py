#!/usr/bin/env python3
"""
安防监控路由
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from src.database.session import get_db
from src.database.models.security import Camera, SecurityAlert, PatrolRoute, PatrolPoint, PatrolRecord
from src.api.models import CameraResponse, CameraCreate, SecurityAlertResponse, SecurityAlertCreate

router = APIRouter(prefix="/security", tags=["安防监控"])


@router.get("/cameras", response_model=list[CameraResponse])
async def list_cameras(db: AsyncSession = Depends(get_db)):
    """获取摄像头列表"""
    result = await db.execute(select(Camera))
    return result.scalars().all()


@router.post("/cameras", response_model=CameraResponse, status_code=status.HTTP_201_CREATED)
async def create_camera(camera_in: CameraCreate, db: AsyncSession = Depends(get_db)):
    """创建摄像头"""
    camera = Camera(
        room_id=camera_in.room_id,
        name=camera_in.name,
        code=camera_in.code,
        ip_address=camera_in.ip_address,
        resolution=camera_in.resolution,
        location_desc=camera_in.location_desc,
        rtsp_url=camera_in.rtsp_url,
    )
    db.add(camera)
    await db.commit()
    await db.refresh(camera)
    return camera


@router.get("/alerts", response_model=list[SecurityAlertResponse])
async def list_alerts(status: str = None, db: AsyncSession = Depends(get_db)):
    """获取安防告警列表"""
    query = select(SecurityAlert)
    if status:
        query = query.where(SecurityAlert.status == status)
    result = await db.execute(query)
    return result.scalars().all()


@router.post("/alerts", response_model=SecurityAlertResponse, status_code=status.HTTP_201_CREATED)
async def create_alert(alert_in: SecurityAlertCreate, db: AsyncSession = Depends(get_db)):
    """创建安防告警"""
    alert = SecurityAlert(
        camera_id=alert_in.camera_id,
        alert_type=alert_in.alert_type,
        level=alert_in.level,
        message=alert_in.message,
        image_url=alert_in.image_url,
    )
    db.add(alert)
    await db.commit()
    await db.refresh(alert)
    return alert


@router.put("/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: int, db: AsyncSession = Depends(get_db)):
    """处理告警"""
    result = await db.execute(select(SecurityAlert).where(SecurityAlert.id == alert_id))
    alert = result.scalar_one_or_none()
    if not alert:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="告警不存在")
    
    alert.status = "resolved"
    await db.commit()
    await db.refresh(alert)
    return alert
