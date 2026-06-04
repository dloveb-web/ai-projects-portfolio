#!/usr/bin/env python3
"""
API数据模型
"""

from pydantic import BaseModel, EmailStr
from typing import Optional, Any
from datetime import datetime


# 用户相关模型
class UserBase(BaseModel):
    """用户基础模型"""
    username: str
    email: EmailStr
    full_name: Optional[str] = None


class UserCreate(UserBase):
    """创建用户模型"""
    password: str


class UserLogin(BaseModel):
    """用户登录模型"""
    username: str
    password: str


class UserResponse(UserBase):
    """用户响应模型"""
    id: int
    is_active: bool
    is_admin: bool
    created_at: datetime
    
    class Config:
        from_attributes = True


class Token(BaseModel):
    """令牌模型"""
    access_token: str
    token_type: str


class TokenData(BaseModel):
    """令牌数据"""
    username: Optional[str] = None


# 园区相关模型
class ParkBase(BaseModel):
    """园区基础模型"""
    name: str
    code: str
    address: Optional[str] = None
    area: Optional[float] = None
    building_count: Optional[int] = None
    description: Optional[str] = None


class ParkCreate(ParkBase):
    """创建园区模型"""
    pass


class ParkResponse(ParkBase):
    """园区响应模型"""
    id: int
    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class BuildingBase(BaseModel):
    """楼栋基础模型"""
    name: str
    code: str
    floor_count: Optional[int] = None
    area: Optional[float] = None
    type: Optional[str] = None
    description: Optional[str] = None


class BuildingCreate(BuildingBase):
    """创建楼栋模型"""
    pass


class BuildingResponse(BuildingBase):
    """楼栋响应模型"""
    id: int
    park_id: int
    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class RoomBase(BaseModel):
    """房间基础模型"""
    name: str
    code: str
    floor: Optional[int] = None
    area: Optional[float] = None
    type: Optional[str] = None
    status: Optional[str] = "available"
    description: Optional[str] = None


class RoomCreate(RoomBase):
    """创建房间模型"""
    pass


class RoomResponse(RoomBase):
    """房间响应模型"""
    id: int
    building_id: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


# 设备相关模型
class DeviceTypeBase(BaseModel):
    """设备类型基础模型"""
    name: str
    category: Optional[str] = None
    description: Optional[str] = None


class DeviceTypeCreate(DeviceTypeBase):
    """创建设备类型模型"""
    pass


class DeviceTypeResponse(DeviceTypeBase):
    """设备类型响应模型"""
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True


class DeviceBase(BaseModel):
    """设备基础模型"""
    name: str
    code: str
    device_type_id: Optional[int] = None
    room_id: Optional[int] = None
    ip_address: Optional[str] = None
    location: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


class DeviceCreate(DeviceBase):
    """创建设备模型"""
    pass


class DeviceResponse(DeviceBase):
    """设备响应模型"""
    id: int
    status: str
    last_heartbeat: Optional[datetime] = None
    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


# 通行管理模型
class PersonBase(BaseModel):
    """人员基础模型"""
    name: str
    id_card: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    person_type: str
    department: Optional[str] = None
    photo_url: Optional[str] = None


class PersonCreate(PersonBase):
    """创建人员模型"""
    pass


class PersonResponse(PersonBase):
    """人员响应模型"""
    id: int
    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class VisitorBase(BaseModel):
    """访客基础模型"""
    name: str
    id_card: Optional[str] = None
    phone: Optional[str] = None
    company: Optional[str] = None
    purpose: Optional[str] = None
    host_id: Optional[int] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


class VisitorCreate(VisitorBase):
    """创建访客模型"""
    pass


class VisitorResponse(VisitorBase):
    """访客响应模型"""
    id: int
    status: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


# 能源管理模型
class EnergyMeterBase(BaseModel):
    """能源计量表基础模型"""
    name: str
    code: str
    room_id: Optional[int] = None
    meter_type: str
    unit: Optional[str] = None
    installed_at: Optional[datetime] = None


class EnergyMeterCreate(EnergyMeterBase):
    """创建能源计量表模型"""
    pass


class EnergyMeterResponse(EnergyMeterBase):
    """能源计量表响应模型"""
    id: int
    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class EnergyConsumptionBase(BaseModel):
    """能耗记录基础模型"""
    meter_id: int
    value: float
    unit: Optional[str] = None


class EnergyConsumptionCreate(EnergyConsumptionBase):
    """创建能耗记录模型"""
    pass


class EnergyConsumptionResponse(EnergyConsumptionBase):
    """能耗记录响应模型"""
    id: int
    timestamp: datetime
    created_at: datetime
    
    class Config:
        from_attributes = True


# 安防监控模型
class CameraBase(BaseModel):
    """摄像头基础模型"""
    name: str
    code: str
    room_id: Optional[int] = None
    ip_address: Optional[str] = None
    resolution: Optional[str] = None
    location_desc: Optional[str] = None
    rtsp_url: Optional[str] = None


class CameraCreate(CameraBase):
    """创建摄像头模型"""
    pass


class CameraResponse(CameraBase):
    """摄像头响应模型"""
    id: int
    status: str
    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class SecurityAlertBase(BaseModel):
    """安防告警基础模型"""
    camera_id: Optional[int] = None
    alert_type: str
    level: str
    message: Optional[str] = None
    image_url: Optional[str] = None


class SecurityAlertCreate(SecurityAlertBase):
    """创建安防告警模型"""
    pass


class SecurityAlertResponse(SecurityAlertBase):
    """安防告警响应模型"""
    id: int
    timestamp: datetime
    status: str
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[int] = None
    
    class Config:
        from_attributes = True
