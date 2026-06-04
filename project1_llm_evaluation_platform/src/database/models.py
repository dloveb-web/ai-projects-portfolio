from sqlalchemy import Column, Integer, String, Text, Float, DateTime, JSON, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from .connection import Base


class EvaluationResult(Base):
    __tablename__ = "evaluation_results"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(100), nullable=False, index=True)
    prompt = Column(Text, nullable=False)
    response = Column(Text)
    metrics = Column(JSON)
    latency_ms = Column(Float)
    cost = Column(Float)
    benchmark_type = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class Prompt(Base):
    __tablename__ = "prompts"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False)
    content = Column(Text, nullable=False)
    version = Column(String(50), default="v1.0")
    category = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ABTest(Base):
    __tablename__ = "ab_tests"

    id = Column(Integer, primary_key=True, index=True)
    prompt_a = Column(Text, nullable=False)
    prompt_b = Column(Text, nullable=False)
    test_prompt = Column(Text, nullable=False)
    result_a = Column(JSON)
    result_b = Column(JSON)
    winner = Column(String(10))
    created_at = Column(DateTime, default=datetime.utcnow)


class SecurityLog(Base):
    __tablename__ = "security_logs"

    id = Column(Integer, primary_key=True, index=True)
    input_text = Column(Text, nullable=False)
    threat_type = Column(String(100))
    threat_level = Column(String(20))
    action = Column(String(50))
    details = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
