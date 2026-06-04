"""
Database models for the AI Smart Butler Robot.
"""
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey, Enum as SQLEnum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
import enum

from src.database import Base
from src.config import PersonalityType, EmotionType


class User(Base):
    """Family member user model."""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    age = Column(Integer)
    role = Column(String(50))  # elderly, adult, child
    
    # Biometric data paths
    face_embedding_path = Column(String(255))
    voiceprint_path = Column(String(255))
    
    # Personality settings
    preferred_personality = Column(SQLEnum(PersonalityType), default=PersonalityType.ELDERLY)
    
    # Health profile
    has_medical_conditions = Column(Boolean, default=False)
    medical_notes = Column(Text)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    reminders = relationship("Reminder", back_populates="user")
    health_records = relationship("HealthRecord", back_populates="user")
    conversations = relationship("Conversation", back_populates="user")


class Reminder(Base):
    """Reminder model for medication, schedules, etc."""
    __tablename__ = "reminders"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    title = Column(String(200), nullable=False)
    description = Column(Text)
    
    reminder_type = Column(String(50))  # medication, schedule, task, etc.
    medication_name = Column(String(100))
    medication_dosage = Column(String(100))
    
    scheduled_time = Column(DateTime(timezone=True), nullable=False)
    repeat_rule = Column(String(100))  # daily, weekly, etc.
    
    is_active = Column(Boolean, default=True)
    last_reminded_at = Column(DateTime(timezone=True))
    confirmed_at = Column(DateTime(timezone=True))
    confirmation_status = Column(String(20), default="pending")  # pending, confirmed, missed, escalated
    
    escalation_level = Column(Integer, default=0)  # 0: none, 1: second reminder, 2: notify family
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="reminders")


class HealthRecord(Base):
    """Health monitoring data model."""
    __tablename__ = "health_records"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    
    # Vital signs
    temperature = Column(Float)
    breathing_rate = Column(Integer)
    heart_rate = Column(Integer)
    blood_pressure_systolic = Column(Integer)
    blood_pressure_diastolic = Column(Integer)
    oxygen_level = Column(Float)
    
    is_abnormal = Column(Boolean, default=False)
    notes = Column(Text)
    
    recorded_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="health_records")


class Conversation(Base):
    """Conversation history model."""
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    
    user_input = Column(Text, nullable=False)
    assistant_response = Column(Text, nullable=False)
    
    # Context
    personality_used = Column(SQLEnum(PersonalityType))
    emotion_detected = Column(SQLEnum(EmotionType))
    
    # Privacy
    is_private = Column(Boolean, default=False)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="conversations")


class EmotionRecord(Base):
    """Emotion detection history model."""
    __tablename__ = "emotion_records"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    
    emotion = Column(SQLEnum(EmotionType), nullable=False)
    confidence = Column(Float)
    
    # Source: facial, voice, combined
    source = Column(String(20))
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class HomeDevice(Base):
    """Smart home device model."""
    __tablename__ = "home_devices"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    device_type = Column(String(50))  # light, ac, tv, etc.
    
    control_type = Column(String(50))  # infrared, wifi, matter
    device_address = Column(String(255))
    
    is_active = Column(Boolean, default=True)
    current_state = Column(Text)  # JSON of current state
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class EmergencyContact(Base):
    """Emergency contact model."""
    __tablename__ = "emergency_contacts"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    phone = Column(String(20), nullable=False)
    relationship = Column(String(50))
    priority = Column(Integer, default=1)  # 1: primary, 2: secondary
    
    is_active = Column(Boolean, default=True)
    notify_on_emergency = Column(Boolean, default=True)
    notify_on_health_anomaly = Column(Boolean, default=False)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class EmergencyEvent(Base):
    """Emergency event log model."""
    __tablename__ = "emergency_events"
    
    id = Column(Integer, primary_key=True, index=True)
    
    event_type = Column(String(50))  # fall, no_activity, sos_button, voice_sos
    severity = Column(String(20))  # low, medium, high, critical
    
    triggered_at = Column(DateTime(timezone=True), server_default=func.now())
    resolved_at = Column(DateTime(timezone=True))
    is_resolved = Column(Boolean, default=False)
    
    notified_contacts = Column(Text)  # JSON list of notified contacts
    notes = Column(Text)
    
    location = Column(String(255))  # if indoor positioning available


class SystemLog(Base):
    """System operation log model."""
    __tablename__ = "system_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    
    level = Column(String(20))  # info, warning, error
    component = Column(String(50))  # llm, voice, vision, etc.
    message = Column(Text)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class PrivacyAudit(Base):
    """Privacy audit log model for compliance."""
    __tablename__ = "privacy_audits"
    
    id = Column(Integer, primary_key=True, index=True)
    
    action_type = Column(String(50))  # data_access, data_deletion, encryption, etc.
    component = Column(String(50))
    is_local_processing = Column(Boolean, default=True)
    data_type = Column(String(50))
    
    notes = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

