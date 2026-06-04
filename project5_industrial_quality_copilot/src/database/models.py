from datetime import datetime, timezone

from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship, declarative_base


def _now_utc() -> datetime:
    """Return current UTC datetime (compatible with Python 3.12+)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(20), default="engineer")
    created_at = Column(DateTime, default=_now_utc)

    tasks = relationship("Task", back_populates="user")


class Task(Base):
    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    status = Column(String(20), default="pending")
    created_at = Column(DateTime, default=_now_utc)

    user = relationship("User", back_populates="tasks")
    detections = relationship("Detection", back_populates="task")
    report = relationship("Report", back_populates="task", uselist=False)


class Detection(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(Integer, ForeignKey("tasks.id"))
    image_path = Column(String(500))
    has_defect = Column(Boolean, default=False)
    processing_time = Column(Float)
    created_at = Column(DateTime, default=_now_utc)

    task = relationship("Task", back_populates="detections")
    defects = relationship("Defect", back_populates="detection")


class Defect(Base):
    __tablename__ = "defects"

    id = Column(Integer, primary_key=True, autoincrement=True)
    detection_id = Column(Integer, ForeignKey("detections.id"))
    class_name = Column(String(100))
    confidence = Column(Float)
    bbox = Column(JSON)
    severity = Column(String(20))

    detection = relationship("Detection", back_populates="defects")
    analysis = relationship("Analysis", back_populates="defect", uselist=False)


class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, autoincrement=True)
    defect_id = Column(Integer, ForeignKey("defects.id"))
    defect_type = Column(String(100))
    severity = Column(String(20))
    cause_analysis = Column(Text)
    suggestions = Column(JSON)
    created_at = Column(DateTime, default=_now_utc)

    defect = relationship("Defect", back_populates="analysis")


class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(Integer, ForeignKey("tasks.id"))
    file_path = Column(String(500))
    summary = Column(JSON)
    generated_at = Column(DateTime, default=_now_utc)

    task = relationship("Task", back_populates="report")


class KnowledgeBase(Base):
    __tablename__ = "knowledge_base"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(200))
    content = Column(Text)
    category = Column(String(100))
    extra_metadata = Column("metadata", JSON)  # "metadata" reserved by SQLAlchemy
    created_at = Column(DateTime, default=_now_utc)
