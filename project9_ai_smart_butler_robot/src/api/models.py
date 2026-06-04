"""
API request/response models for the AI Smart Butler Robot.
"""
from pydantic import BaseModel, ConfigDict
from typing import List, Optional, Dict, Any
from datetime import datetime
from src.config import PersonalityType, EmotionType


class HealthCheckResponse(BaseModel):
    """Health check response."""
    status: str
    local_llm: bool
    personality: PersonalityType


# User APIs
class UserCreateRequest(BaseModel):
    """Request to create a new user."""
    name: str
    age: Optional[int] = None
    role: Optional[str] = None
    preferred_personality: PersonalityType = PersonalityType.ELDERLY
    medical_notes: Optional[str] = None


class UserResponse(BaseModel):
    """User response model."""
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    name: str
    age: Optional[int] = None
    role: Optional[str] = None
    preferred_personality: PersonalityType
    created_at: datetime


# Reminder APIs
class ReminderCreateRequest(BaseModel):
    """Request to create a new reminder."""
    user_id: int
    title: str
    description: Optional[str] = None
    reminder_type: str = "schedule"
    medication_name: Optional[str] = None
    medication_dosage: Optional[str] = None
    scheduled_time: datetime
    repeat_rule: Optional[str] = None


class ReminderResponse(BaseModel):
    """Reminder response model."""
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    user_id: int
    title: str
    description: Optional[str] = None
    reminder_type: str
    scheduled_time: datetime
    confirmation_status: str
    escalation_level: int
    created_at: datetime


class ReminderConfirmRequest(BaseModel):
    """Request to confirm a reminder."""
    reminder_id: int
    confirmed: bool
    notes: Optional[str] = None


# Chat APIs
class ChatRequest(BaseModel):
    """Chat request model."""
    user_id: Optional[int] = None
    message: str
    use_personality: Optional[PersonalityType] = None


class ChatResponse(BaseModel):
    """Chat response model."""
    response: str
    personality_used: PersonalityType
    emotion_detected: Optional[EmotionType] = None


# Health APIs
class HealthRecordRequest(BaseModel):
    """Request to create a health record."""
    user_id: int
    temperature: Optional[float] = None
    breathing_rate: Optional[int] = None
    heart_rate: Optional[int] = None
    blood_pressure_systolic: Optional[int] = None
    blood_pressure_diastolic: Optional[int] = None
    oxygen_level: Optional[float] = None
    notes: Optional[str] = None


class HealthRecordResponse(BaseModel):
    """Health record response model."""
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    user_id: int
    temperature: Optional[float] = None
    breathing_rate: Optional[int] = None
    heart_rate: Optional[int] = None
    is_abnormal: bool
    recorded_at: datetime


# Home Control APIs
class HomeDeviceControlRequest(BaseModel):
    """Request to control a home device."""
    device_id: int
    action: str  # on, off, set_temperature, etc.
    parameters: Optional[Dict[str, Any]] = None


class HomeDeviceResponse(BaseModel):
    """Home device response model."""
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    name: str
    device_type: str
    is_active: bool
    current_state: Optional[str] = None


# Emergency APIs
class EmergencyTriggerRequest(BaseModel):
    """Request to trigger emergency manually."""
    event_type: str = "manual_sos"
    notes: Optional[str] = None


class EmergencyResponse(BaseModel):
    """Emergency event response."""
    event_id: int
    event_type: str
    severity: str
    triggered_at: datetime
    is_resolved: bool
    message: str


# Personality APIs
class PersonalitySwitchRequest(BaseModel):
    """Request to switch personality."""
    user_id: Optional[int] = None
    personality: PersonalityType


class PersonalityResponse(BaseModel):
    """Personality info response."""
    current_personality: PersonalityType
    personality_name: str
    available_personalities: List[Dict[str, str]]

