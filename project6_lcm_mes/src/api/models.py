from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


class UserBase(BaseModel):
    username: str
    role: str
    department: Optional[str] = None
    name: Optional[str] = None


class UserCreate(UserBase):
    password: str


class UserResponse(UserBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str


class LoginRequest(BaseModel):
    username: str
    password: str


class ProductionOrderBase(BaseModel):
    order_no: str
    product_code: str
    product_name: str
    quantity: int
    delivery_date: Optional[datetime] = None


class ProductionOrderResponse(ProductionOrderBase):
    id: int
    status: str
    created_at: datetime

    class Config:
        from_attributes = True


class WorkOrderBase(BaseModel):
    wo_no: str
    product_code: str
    product_name: str
    quantity: int
    equipment_id: Optional[int] = None


class WorkOrderCreate(WorkOrderBase):
    order_id: Optional[int] = None


class WorkOrderResponse(WorkOrderBase):
    id: int
    order_id: Optional[int] = None
    completed_quantity: int
    status: str
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    created_at: datetime

    class Config:
        from_attributes = True


class ProductTraceResponse(BaseModel):
    code_8bit: Optional[str]
    code_31bit: Optional[str]
    code_71bit: Optional[str]
    panel_lot: Optional[str]
    polarizer_lot: Optional[str]
    reflector_lot: Optional[str]
    driver_ic_lot: Optional[str]
    fpc_lot: Optional[str]
    glass_cover_lot: Optional[str]
    backlight_lot: Optional[str]
    current_status: Optional[str]


class DefectRecordBase(BaseModel):
    product_code: str
    defect_type: Optional[str] = None
    defect_code: Optional[str] = None
    severity: Optional[str] = None
    description: Optional[str] = None
    location: Optional[str] = None


class RepairRecordBase(BaseModel):
    defect_id: int
    repair_type: Optional[str] = None
    repair_description: Optional[str] = None


class EquipmentBase(BaseModel):
    code: str
    name: str
    type: Optional[str] = None
    model: Optional[str] = None
    location: Optional[str] = None


class EquipmentResponse(EquipmentBase):
    id: int
    status: str
    health_score: int
    last_maintenance: Optional[datetime] = None
    created_at: datetime

    class Config:
        from_attributes = True


class MaterialBatchBase(BaseModel):
    batch_no: str
    material_id: int
    quantity: float
    supplier: Optional[str] = None


class BoxRecordBase(BaseModel):
    box_code: str
    work_order_id: int
    product_list: str
    quantity: int


class DashboardStats(BaseModel):
    total_orders: int
    active_work_orders: int
    total_equipment: int
    running_equipment: int
    today_production: int
    defect_rate: float
