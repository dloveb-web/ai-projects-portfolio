#!/usr/bin/env python3
"""
园区管理路由
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from src.database.session import get_db
from src.database.models.park import Park, Building, Room
from src.api.models import ParkResponse, ParkCreate, BuildingResponse, BuildingCreate, RoomResponse, RoomCreate

router = APIRouter(prefix="/park", tags=["园区管理"])


@router.get("/", response_model=list[ParkResponse])
async def list_parks(db: AsyncSession = Depends(get_db)):
    """获取园区列表"""
    result = await db.execute(select(Park))
    return result.scalars().all()


@router.get("/{park_id}", response_model=ParkResponse)
async def get_park(park_id: int, db: AsyncSession = Depends(get_db)):
    """获取单个园区"""
    result = await db.execute(select(Park).where(Park.id == park_id))
    park = result.scalar_one_or_none()
    if not park:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="园区不存在")
    return park


@router.post("/", response_model=ParkResponse, status_code=status.HTTP_201_CREATED)
async def create_park(park_in: ParkCreate, db: AsyncSession = Depends(get_db)):
    """创建园区"""
    park = Park(
        name=park_in.name,
        code=park_in.code,
        address=park_in.address,
        area=park_in.area,
        building_count=park_in.building_count,
        description=park_in.description,
    )
    db.add(park)
    await db.commit()
    await db.refresh(park)
    return park


@router.get("/{park_id}/buildings", response_model=list[BuildingResponse])
async def list_buildings(park_id: int, db: AsyncSession = Depends(get_db)):
    """获取园区楼栋列表"""
    result = await db.execute(select(Building).where(Building.park_id == park_id))
    return result.scalars().all()


@router.post("/{park_id}/buildings", response_model=BuildingResponse, status_code=status.HTTP_201_CREATED)
async def create_building(park_id: int, building_in: BuildingCreate, db: AsyncSession = Depends(get_db)):
    """创建楼栋"""
    building = Building(
        park_id=park_id,
        name=building_in.name,
        code=building_in.code,
        floor_count=building_in.floor_count,
        area=building_in.area,
        type=building_in.type,
        description=building_in.description,
    )
    db.add(building)
    await db.commit()
    await db.refresh(building)
    return building


@router.get("/{park_id}/buildings/{building_id}/rooms", response_model=list[RoomResponse])
async def list_rooms(park_id: int, building_id: int, db: AsyncSession = Depends(get_db)):
    """获取楼栋房间列表"""
    result = await db.execute(select(Room).where(Room.building_id == building_id))
    return result.scalars().all()


@router.post("/{park_id}/buildings/{building_id}/rooms", response_model=RoomResponse, status_code=status.HTTP_201_CREATED)
async def create_room(park_id: int, building_id: int, room_in: RoomCreate, db: AsyncSession = Depends(get_db)):
    """创建房间"""
    room = Room(
        building_id=building_id,
        name=room_in.name,
        code=room_in.code,
        floor=room_in.floor,
        area=room_in.area,
        type=room_in.type,
        status=room_in.status,
        description=room_in.description,
    )
    db.add(room)
    await db.commit()
    await db.refresh(room)
    return room
