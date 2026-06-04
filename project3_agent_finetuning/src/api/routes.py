from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timezone

from .models import (
    FinetuneTaskCreate, FinetuneTaskResponse,
    FinetunedModelResponse,
    EvaluateRequest, EvaluateResponse,
    AgentCreate, AgentChatRequest, AgentChatResponse,
    AgentSessionResponse, AgentInteractionResponse
)
from ..database import get_db, SessionLocal
from ..database.models import FinetuneTask, FinetunedModel, EvaluationResult, AgentSession, AgentInteraction
from ..core.finetune import finetune_manager
from ..core.agent import agent_manager
from ..core.evaluate import evaluator

router = APIRouter(prefix="/api/v1")


def run_finetune_task(task_id: int):
    """后台运行微调任务 - 内部创建独立的数据库会话
    
    注意：后台任务需要自行创建session，不能使用请求级别的session，
    因为FastAPI会在请求结束后关闭该session。
    """
    db = SessionLocal()
    try:
        task_result = finetune_manager.start_task(task_id)

        db_task = db.query(FinetuneTask).filter(FinetuneTask.id == task_id).first()
        if db_task:
            db_task.status = task_result["status"]
            db_task.metrics = task_result.get("metrics", {})
            if task_result["status"] == "completed":
                db_task.completed_at = datetime.now(timezone.utc)

                model = FinetunedModel(
                    task_id=task_id,
                    model_path=task_result.get("model_path", ""),
                    version="v1.0",
                    performance=task_result.get("metrics", {})
                )
                db.add(model)

            db.commit()
    finally:
        db.close()


@router.post("/finetune/create", response_model=FinetuneTaskResponse)
def create_finetune_task(
    request: FinetuneTaskCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    db_task = FinetuneTask(
        model_name=request.model_name,
        method=request.method,
        status="pending",
        config=request.config,
        train_data_path=request.train_data_path
    )
    db.add(db_task)
    db.commit()
    db.refresh(db_task)

    finetune_manager.create_task(
        db_task.id,
        request.model_name,
        request.method,
        request.config
    )

    return FinetuneTaskResponse(
        id=db_task.id,
        model_name=db_task.model_name,
        method=db_task.method,
        status=db_task.status,
        config=db_task.config or {},
        metrics=db_task.metrics or {},
        created_at=db_task.created_at,
        completed_at=db_task.completed_at
    )


@router.post("/finetune/tasks/{task_id}/start")
def start_finetune_task(
    task_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    db_task = db.query(FinetuneTask).filter(FinetuneTask.id == task_id).first()
    if not db_task:
        raise HTTPException(status_code=404, detail="Task not found")

    # 注意：不再传递db参数，后台任务内部创建独立的session
    background_tasks.add_task(run_finetune_task, task_id)

    db_task.status = "running"
    db.commit()

    return {"message": "Task started", "task_id": task_id}


@router.get("/finetune/tasks", response_model=List[FinetuneTaskResponse])
def list_finetune_tasks(db: Session = Depends(get_db)):
    tasks = db.query(FinetuneTask).order_by(FinetuneTask.created_at.desc()).all()
    return [
        FinetuneTaskResponse(
            id=t.id,
            model_name=t.model_name,
            method=t.method,
            status=t.status,
            config=t.config or {},
            metrics=t.metrics or {},
            created_at=t.created_at,
            completed_at=t.completed_at
        )
        for t in tasks
    ]


@router.get("/finetune/tasks/{task_id}", response_model=FinetuneTaskResponse)
def get_finetune_task(task_id: int, db: Session = Depends(get_db)):
    task = db.query(FinetuneTask).filter(FinetuneTask.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    mem_task = finetune_manager.get_task_status(task_id)
    if mem_task and mem_task.get("status") != task.status:
        task.status = mem_task["status"]
        task.metrics = mem_task.get("metrics", {})
        db.commit()

    return FinetuneTaskResponse(
        id=task.id,
        model_name=task.model_name,
        method=task.method,
        status=task.status,
        config=task.config or {},
        metrics=task.metrics or {},
        created_at=task.created_at,
        completed_at=task.completed_at
    )


@router.get("/finetune/methods")
def get_finetune_methods():
    return {"methods": finetune_manager.get_available_methods()}


@router.post("/finetune/compare")
def compare_methods(model_name: str):
    results = finetune_manager.compare_methods(model_name, [])
    return results


@router.get("/models", response_model=List[FinetunedModelResponse])
def list_models(db: Session = Depends(get_db)):
    models = db.query(FinetunedModel).order_by(FinetunedModel.created_at.desc()).all()
    return [
        FinetunedModelResponse(
            id=m.id,
            task_id=m.task_id,
            model_path=m.model_path,
            version=m.version,
            performance=m.performance or {},
            created_at=m.created_at
        )
        for m in models
    ]


@router.post("/evaluate", response_model=EvaluateResponse)
def evaluate_model(request: EvaluateRequest, db: Session = Depends(get_db)):
    result = evaluator.evaluate_model(request.model_id, request.task_type, request.config)

    db_result = EvaluationResult(
        model_id=request.model_id,
        task_type=request.task_type,
        metrics=result["metrics"]
    )
    db.add(db_result)
    db.commit()

    return EvaluateResponse(
        task_type=result["task_type"],
        metrics=result["metrics"],
        created_at=datetime.utcnow()
    )


@router.get("/evaluate/reports")
def get_evaluation_reports(model_id: Optional[int] = None, db: Session = Depends(get_db)):
    query = db.query(EvaluationResult)
    if model_id:
        query = query.filter(EvaluationResult.model_id == model_id)
    results = query.order_by(EvaluationResult.created_at.desc()).all()
    return {"reports": [
        {
            "id": r.id,
            "model_id": r.model_id,
            "task_type": r.task_type,
            "metrics": r.metrics,
            "created_at": r.created_at
        }
        for r in results
    ]}


@router.post("/agent/create", response_model=AgentSessionResponse)
def create_agent_session(request: AgentCreate, db: Session = Depends(get_db)):
    db_session = AgentSession(
        agent_type=request.agent_type,
        config=request.config
    )
    db.add(db_session)
    db.commit()
    db.refresh(db_session)

    agent_manager.create_session(
        db_session.id,
        request.agent_type,
        request.config
    )

    return AgentSessionResponse(
        id=db_session.id,
        agent_type=db_session.agent_type,
        config=db_session.config or {},
        created_at=db_session.created_at
    )


@router.post("/agent/chat", response_model=AgentChatResponse)
def agent_chat(request: AgentChatRequest, db: Session = Depends(get_db)):
    try:
        result = agent_manager.chat(request.session_id, request.user_input, request.image)

        db_interaction = AgentInteraction(
            session_id=request.session_id,
            user_input=request.user_input,
            agent_output=result["output"],
            tools_used=result.get("tools_used", [])
        )
        db.add(db_interaction)
        db.commit()

        return AgentChatResponse(
            success=True,
            output=result["output"],
            tools_used=result.get("tools_used", [])
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/agent/sessions", response_model=List[AgentSessionResponse])
def list_agent_sessions(db: Session = Depends(get_db)):
    sessions = db.query(AgentSession).order_by(AgentSession.created_at.desc()).all()
    return [
        AgentSessionResponse(
            id=s.id,
            agent_type=s.agent_type,
            config=s.config or {},
            created_at=s.created_at
        )
        for s in sessions
    ]


@router.get("/agent/sessions/{session_id}/history", response_model=List[AgentInteractionResponse])
def get_agent_history(session_id: int, db: Session = Depends(get_db)):
    interactions = db.query(AgentInteraction).filter(
        AgentInteraction.session_id == session_id
    ).order_by(AgentInteraction.timestamp).all()

    return [
        AgentInteractionResponse(
            id=i.id,
            session_id=i.session_id,
            user_input=i.user_input,
            agent_output=i.agent_output,
            tools_used=i.tools_used or [],
            timestamp=i.timestamp
        )
        for i in interactions
    ]


@router.get("/agent/types")
def get_agent_types():
    return {"types": agent_manager.get_available_agents()}


@router.get("/health")
def health_check():
    return {"status": "healthy", "service": "Agent Finetune Platform"}

