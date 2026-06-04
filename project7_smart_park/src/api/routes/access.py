#!/usr/bin/env python3
"""
通行管理路由
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from src.database.session import get_db
from src.database.models.access import Person, Visitor, AccessRecord
from src.api.models import PersonResponse, PersonCreate, VisitorResponse, VisitorCreate

router = APIRouter(prefix="/access", tags=["通行管理"])


@router.get("/persons", response_model=list[PersonResponse])
async def list_persons(db: AsyncSession = Depends(get_db)):
    """获取人员列表"""
    result = await db.execute(select(Person))
    return result.scalars().all()


@router.post("/persons", response_model=PersonResponse, status_code=status.HTTP_201_CREATED)
async def create_person(person_in: PersonCreate, db: AsyncSession = Depends(get_db)):
    """创建人员"""
    person = Person(
        name=person_in.name,
        id_card=person_in.id_card,
        phone=person_in.phone,
        email=person_in.email,
        person_type=person_in.person_type,
        department=person_in.department,
        photo_url=person_in.photo_url,
    )
    db.add(person)
    await db.commit()
    await db.refresh(person)
    return person


@router.get("/visitors", response_model=list[VisitorResponse])
async def list_visitors(db: AsyncSession = Depends(get_db)):
    """获取访客列表"""
    result = await db.execute(select(Visitor))
    return result.scalars().all()


@router.post("/visitors", response_model=VisitorResponse, status_code=status.HTTP_201_CREATED)
async def create_visitor(visitor_in: VisitorCreate, db: AsyncSession = Depends(get_db)):
    """创建访客"""
    visitor = Visitor(
        name=visitor_in.name,
        id_card=visitor_in.id_card,
        phone=visitor_in.phone,
        company=visitor_in.company,
        purpose=visitor_in.purpose,
        host_id=visitor_in.host_id,
        start_time=visitor_in.start_time,
        end_time=visitor_in.end_time,
    )
    db.add(visitor)
    await db.commit()
    await db.refresh(visitor)
    return visitor


@router.put("/visitors/{visitor_id}/approve")
async def approve_visitor(visitor_id: int, db: AsyncSession = Depends(get_db)):
    """审核访客"""
    result = await db.execute(select(Visitor).where(Visitor.id == visitor_id))
    visitor = result.scalar_one_or_none()
    if not visitor:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="访客不存在")
    
    visitor.status = "approved"
    await db.commit()
    await db.refresh(visitor)
    return visitor
