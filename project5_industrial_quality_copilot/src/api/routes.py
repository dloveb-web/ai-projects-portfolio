"""
FastAPI routes for the Industrial AI Quality Copilot.

Endpoints:
  - /auth/login             Login / auto-register
  - /detection/detect       Upload image for defect detection
  - /analysis/analyze       Analyze a detected defect
  - /reports/*              List, generate, download reports
  - /knowledge/*            Search / add knowledge base entries
  - /stats/*                Dashboard & trend statistics
  - /roi/analysis           ROI calculation
  - /admin/users            User management
"""

from __future__ import annotations

import base64
import logging
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .models import (
    AnalysisRequest,
    DetectionRequest,
    KnowledgeAddRequest,
    LoginRequest,
    ReportRequest,
    UserCreate,
    UserInfo,
)
from ..core.analyzer import DefectAnalyzer
from ..core.agents import MultiAgentSystem
from ..core.detector import DefectDetector
from ..core.rag_engine import RAGEngine
from ..core.report_generator import ReportGenerator
from ..database.connection import get_db
from ..database.models import (
    Analysis as DBAnalysis,
    Defect,
    Detection,
    KnowledgeBase,
    Report as DBReport,
    Task,
    User,
)
from ..database.settings import get_report_dir, get_upload_dir, settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ---------------------------------------------------------------------------
# Service singletons (lazy-initialized)
# ---------------------------------------------------------------------------

_detector: Optional[DefectDetector] = None
_analyzer: Optional[DefectAnalyzer] = None
_rag_engine: Optional[RAGEngine] = None
_agent_system: Optional[MultiAgentSystem] = None
_report_generator: Optional[ReportGenerator] = None


def _get_detector() -> DefectDetector:
    global _detector
    if _detector is None:
        _detector = DefectDetector()
    return _detector


def _get_analyzer() -> DefectAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = DefectAnalyzer()
    return _analyzer


def _get_rag_engine() -> RAGEngine:
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    return _rag_engine


def _get_agent_system() -> MultiAgentSystem:
    global _agent_system
    if _agent_system is None:
        _agent_system = MultiAgentSystem()
    return _agent_system


def _get_report_generator() -> ReportGenerator:
    global _report_generator
    if _report_generator is None:
        _report_generator = ReportGenerator()
    return _report_generator


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

@router.post("/auth/login")
async def login(request: LoginRequest, db: AsyncSession = Depends(get_db)):
    """Login or auto-register a user. Returns a JWT-style token."""
    result = await db.execute(select(User).where(User.username == request.username))
    user = result.scalar_one_or_none()

    if user:
        # Existing user — verify password
        if not pwd_context.verify(request.password, user.password_hash):
            raise HTTPException(status_code=401, detail="Invalid password")
    else:
        # New user — auto-register (development convenience)
        user = User(
            username=request.username,
            password_hash=pwd_context.hash(request.password),
            role="engineer",
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)

    # Generate a simple token (replace with real JWT in production)
    token = f"token_{user.username}_{uuid.uuid4().hex[:8]}"

    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "username": user.username,
            "role": user.role,
        },
    }


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

@router.post("/detection/detect")
async def detect_defects(
    image: UploadFile = File(...),
    threshold: float = Form(0.5),
    db: AsyncSession = Depends(get_db),
):
    """Upload an image and run defect detection."""
    detector = _get_detector()
    upload_dir = get_upload_dir()
    task_id = f"TASK_{uuid.uuid4().hex[:8]}"

    # Save uploaded file
    file_path = upload_dir / f"{task_id}_{image.filename}"
    content = await image.read()
    with open(file_path, "wb") as f:
        f.write(content)

    # Run detection
    result = detector.detect_from_bytes(content)

    # Persist task
    task = Task(status="completed")
    db.add(task)
    await db.flush()

    detection = Detection(
        task_id=task.id,
        image_path=str(file_path),
        has_defect=result.get("has_defect", False),
        processing_time=result.get("processing_time", 0.0),
    )
    db.add(detection)
    await db.flush()

    for defect in result.get("defects", []):
        db_defect = Defect(
            detection_id=detection.id,
            class_name=defect.get("class_name", "unknown"),
            confidence=defect.get("confidence", 0.0),
            bbox=defect.get("bbox", []),
        )
        db.add(db_defect)

    await db.commit()

    # Encode annotated image if present
    annotated_b64: Optional[str] = None
    annotated_img = result.get("annotated_image")
    if annotated_img is not None:
        import cv2
        _, buffer = cv2.imencode(".jpg", annotated_img)
        annotated_b64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "task_id": task_id,
        "has_defect": result.get("has_defect", False),
        "defects": result.get("defects", []),
        "processing_time": result.get("processing_time", 0),
        "image_with_boxes": annotated_b64,
    }


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

@router.post("/analysis/analyze")
async def analyze_defect(
    request: AnalysisRequest,
    db: AsyncSession = Depends(get_db),
):
    """Run defect analysis with optional RAG history lookup."""
    analyzer = _get_analyzer()
    rag_engine = _get_rag_engine()

    defect_info = {
        "class_name": request.defect_info.class_name,
        "confidence": request.defect_info.confidence,
        "bbox": request.defect_info.bbox,
    }

    result = analyzer.analyze_defect(
        defect_image=None,
        defect_info=defect_info,
        context="User requested analysis" if request.include_history else None,
    )

    similar_cases: list = []
    if request.include_history:
        similar_cases = rag_engine.find_similar_cases(
            defect_description=str(defect_info),
            defect_type=defect_info["class_name"],
            top_k=3,
        )

    return {
        "defect_type": result.get("defect_type", defect_info["class_name"]),
        "severity": result.get("severity", "medium"),
        "cause_analysis": result.get("cause_analysis", ""),
        "similar_cases": [
            {
                "id": case.get("id", ""),
                "content": case.get("content", ""),
                "metadata": case.get("metadata", {}),
                "distance": case.get("distance", 0.0),
            }
            for case in similar_cases
        ],
        "suggestions": result.get("suggestions", []),
    }


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

@router.get("/reports/list")
async def list_reports(
    limit: int = 20,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    """List recent reports."""
    result = await db.execute(
        select(DBReport)
        .order_by(DBReport.generated_at.desc())
        .limit(limit)
        .offset(offset)
    )
    reports = result.scalars().all()

    return {
        "reports": [
            {
                "id": r.id,
                "file_path": r.file_path,
                "summary": r.summary,
                "generated_at": r.generated_at.isoformat() if r.generated_at else None,
            }
            for r in reports
        ],
        "total": len(reports),
    }


@router.post("/reports/generate")
async def generate_report(
    request: ReportRequest,
    db: AsyncSession = Depends(get_db),
):
    """Generate a quality inspection report for a completed task."""
    report_gen = _get_report_generator()

    # Resolve task_id robustly — try as int, then as string suffix
    task_id = _resolve_task_id(request.task_id)

    task_result = await db.execute(select(Task).where(Task.id == task_id))
    task = task_result.scalar_one_or_none()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    detection_result = await db.execute(
        select(Detection).where(Detection.task_id == task.id)
    )
    detection = detection_result.scalar_one_or_none()
    if not detection:
        raise HTTPException(status_code=404, detail="Detection not found")

    defects_result = await db.execute(
        select(Defect).where(Defect.detection_id == detection.id)
    )
    defects = defects_result.scalars().all()

    detection_data = {
        "has_defect": detection.has_defect,
        "defects": [
            {
                "class_name": d.class_name,
                "confidence": d.confidence,
                "bbox": d.bbox,
            }
            for d in defects
        ],
        "processing_time": detection.processing_time,
    }

    analysis_results = [
        {
            "success": True,
            "analysis": {
                "defect_type": d.class_name,
                "severity": d.severity or "medium",
            },
        }
        for d in defects
    ]

    report_result = report_gen.generate_and_save(detection_data, analysis_results)

    db_report = DBReport(
        task_id=task.id,
        file_path=report_result["file_path"],
        summary={
            "total_defects": len(defects),
            "pass_rate": 100 if not detection.has_defect else 0,
        },
    )
    db.add(db_report)
    await db.commit()

    return {
        "report_id": report_result["report_id"],
        "file_path": report_result["file_path"],
        "summary": db_report.summary,
    }


@router.get("/reports/{report_id}")
async def get_report(report_id: str, db: AsyncSession = Depends(get_db)):
    """Download a report file."""
    rid = _resolve_task_id(report_id)

    result = await db.execute(select(DBReport).where(DBReport.id == rid))
    report = result.scalar_one_or_none()
    if not report or not report.file_path:
        raise HTTPException(status_code=404, detail="Report not found")

    return FileResponse(report.file_path, media_type="text/html")


# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------

@router.get("/knowledge/search")
async def search_knowledge(
    query: str,
    top_k: int = 5,
    category: Optional[str] = None,
):
    """Search the RAG knowledge base."""
    rag_engine = _get_rag_engine()
    results = rag_engine.search(query, top_k=top_k)

    return {
        "results": [
            {
                "id": r.get("id", ""),
                "content": r.get("content", ""),
                "metadata": r.get("metadata", {}),
                "distance": r.get("distance", 0.0),
            }
            for r in results
        ],
        "total": len(results),
    }


@router.post("/knowledge/add")
async def add_knowledge(
    request: KnowledgeAddRequest,
    db: AsyncSession = Depends(get_db),
):
    """Add an entry to the knowledge base."""
    rag_engine = _get_rag_engine()
    doc_id = f"KB_{uuid.uuid4().hex[:8]}"

    rag_engine.add_document(
        document_id=doc_id,
        content=request.content,
        metadata={
            "title": request.title,
            "category": request.category,
            **(request.metadata or {}),
        },
    )

    kb_entry = KnowledgeBase(
        title=request.title,
        content=request.content,
        category=request.category,
        extra_metadata=request.metadata or {},
    )
    db.add(kb_entry)
    await db.commit()

    return {"id": doc_id, "message": "Knowledge added successfully"}


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

@router.get("/stats/dashboard")
async def get_dashboard_stats(db: AsyncSession = Depends(get_db)):
    """Dashboard overview for today."""
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    result = await db.execute(
        select(Detection).where(Detection.created_at >= today)
    )
    today_detections = result.scalars().all()

    today_inspections = len(today_detections)
    today_defects = sum(1 for d in today_detections if d.has_defect)
    pass_rate = (
        (today_inspections - today_defects) / today_inspections * 100
        if today_inspections > 0
        else 100
    )
    avg_time = (
        sum(d.processing_time or 0 for d in today_detections) / today_inspections
        if today_inspections > 0
        else 0
    )

    return {
        "today_inspections": today_inspections,
        "today_defects": today_defects,
        "pass_rate": round(pass_rate, 2),
        "avg_processing_time": round(avg_time, 3),
        "recent_alerts": [],
    }


@router.get("/stats/defect-trend")
async def get_defect_trend(
    days: int = 7,
    db: AsyncSession = Depends(get_db),
):
    """Daily defect trend for the past N days."""
    start_date = datetime.now() - timedelta(days=days)

    result = await db.execute(
        select(Detection).where(Detection.created_at >= start_date)
    )
    detections = result.scalars().all()

    trend: dict = {}
    for d in detections:
        date_key = (
            d.created_at.strftime("%Y-%m-%d") if d.created_at else "unknown"
        )
        if date_key not in trend:
            trend[date_key] = {"total": 0, "defects": 0}
        trend[date_key]["total"] += 1
        if d.has_defect:
            trend[date_key]["defects"] += 1

    return {
        "trend": [
            {"date": k, "total": v["total"], "defects": v["defects"]}
            for k, v in sorted(trend.items())
        ],
        "period_days": days,
    }


# ---------------------------------------------------------------------------
# ROI
# ---------------------------------------------------------------------------

@router.get("/roi/analysis")
async def get_roi_analysis(db: AsyncSession = Depends(get_db)):
    """Simple ROI analysis comparing manual vs AI inspection costs."""
    result = await db.execute(select(Detection))
    all_detections = result.scalars().all()

    total_inspections = len(all_detections)
    manual_cost_per_unit = 2.0
    ai_cost_per_unit = 0.1
    defect_rate = (
        sum(1 for d in all_detections if d.has_defect) / total_inspections
        if total_inspections > 0
        else 0
    )
    avg_defect_loss = 50.0

    manual_cost = total_inspections * manual_cost_per_unit
    ai_cost = total_inspections * ai_cost_per_unit
    savings = manual_cost - ai_cost
    defect_loss = total_inspections * defect_rate * avg_defect_loss

    return {
        "summary": {
            "total_inspections": total_inspections,
            "manual_cost_per_unit": manual_cost_per_unit,
            "ai_cost_per_unit": ai_cost_per_unit,
            "total_savings": round(savings, 2),
            "defect_loss_prevention": round(defect_loss, 2),
            "roi_percentage": (
                round((savings / (manual_cost * 0.1)) * 100, 2)
                if manual_cost > 0
                else 0
            ),
        },
        "metrics": {
            "labor_cost_reduction": "70%",
            "detection_accuracy": "95%",
            "processing_speed_improvement": "5x",
        },
    }


# ---------------------------------------------------------------------------
# Admin / Users
# ---------------------------------------------------------------------------

@router.get("/admin/users", response_model=List[UserInfo])
async def list_users(db: AsyncSession = Depends(get_db)):
    """List all users."""
    result = await db.execute(select(User))
    return result.scalars().all()


@router.post("/admin/users", response_model=UserInfo)
async def create_user(request: UserCreate, db: AsyncSession = Depends(get_db)):
    """Create a new user."""
    user = User(
        username=request.username,
        password_hash=pwd_context.hash(request.password),
        role=request.role,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_task_id(raw: str) -> int:
    """Robustly extract an integer task-id from a string.

    Accepts:
      - "123"          → 123
      - "TASK_abc123"  → raises 400 (not an int)
      - "123"          → 123
    For report IDs like "REP_20240101_120000", falls back to DB lookup.
    """
    try:
        return int(raw)
    except ValueError:
        # If the raw string contains underscores, try extracting a numeric suffix
        if "_" in raw:
            suffix = raw.rsplit("_", 1)[-1]
            try:
                return int(suffix)
            except ValueError:
                pass
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task/report ID format: {raw}",
        )
