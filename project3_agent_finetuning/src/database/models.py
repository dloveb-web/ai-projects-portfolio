from sqlalchemy import Column, Integer, String, Text, Float, DateTime, JSON, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from .connection import Base


class FinetuneTask(Base):
    __tablename__ = "finetune_tasks"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(100), nullable=False, index=True)
    method = Column(String(50), nullable=False)
    status = Column(String(50), default="pending")
    config = Column(JSON)
    metrics = Column(JSON)
    train_data_path = Column(String(500))
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    completed_at = Column(DateTime)

    finetuned_models = relationship("FinetunedModel", back_populates="task")


class FinetunedModel(Base):
    __tablename__ = "finetuned_models"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("finetune_tasks.id"))
    model_path = Column(String(500), nullable=False)
    version = Column(String(50), default="v1.0")
    performance = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

    task = relationship("FinetuneTask", back_populates="finetuned_models")
    evaluation_results = relationship("EvaluationResult", back_populates="model")


class EvaluationResult(Base):
    __tablename__ = "evaluation_results"

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey("finetuned_models.id"))
    task_type = Column(String(50), nullable=False)
    metrics = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

    model = relationship("FinetunedModel", back_populates="evaluation_results")


class AgentSession(Base):
    __tablename__ = "agent_sessions"

    id = Column(Integer, primary_key=True, index=True)
    agent_type = Column(String(50), nullable=False)
    config = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

    interactions = relationship("AgentInteraction", back_populates="session")


class AgentInteraction(Base):
    __tablename__ = "agent_interactions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("agent_sessions.id"))
    user_input = Column(Text, nullable=False)
    agent_output = Column(Text)
    tools_used = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    session = relationship("AgentSession", back_populates="interactions")

