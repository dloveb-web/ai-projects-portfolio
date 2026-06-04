from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from .models import (
    ChatRequest, ChatResponse,
    EvaluateRequest, EvaluateResponse,
    CompareModelsRequest,
    SecurityCheckRequest, SecurityCheckResponse,
    PromptRequest, ABTestRequest,
    ReportRequest
)
from ..database import get_db
from ..core import (
    LLMClient, Evaluator, SecurityChecker,
    PromptLab, ReportGenerator, list_all_models
)

router = APIRouter(prefix="/api/v1")


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, db: Session = Depends(get_db)):
    client = LLMClient(request.model)
    result = client.chat(request.prompt, request.system_prompt, request.temperature)
    return ChatResponse(**result)


@router.get("/models")
def get_models():
    return {"models": list_all_models()}


@router.post("/evaluate", response_model=EvaluateResponse)
def evaluate(request: EvaluateRequest, db: Session = Depends(get_db)):
    evaluator = Evaluator(db)
    result = evaluator.evaluate_model(
        request.model,
        request.benchmark_type,
        request.temperature,
        request.save_results
    )
    return EvaluateResponse(
        model=result["model"],
        benchmark_type=result["benchmark_type"],
        total_questions=result["total_questions"],
        success_count=result["success_count"],
        success_rate=result["success_rate"],
        avg_latency_ms=result["avg_latency_ms"],
        total_cost=result["total_cost"],
        avg_score=result["avg_score"],
        category_scores=result["category_scores"]
    )


@router.post("/compare")
def compare_models(request: CompareModelsRequest, db: Session = Depends(get_db)):
    evaluator = Evaluator(db)
    result = evaluator.compare_models(
        request.models,
        request.benchmark_type,
        request.temperature
    )
    return result


@router.post("/security/check", response_model=SecurityCheckResponse)
def security_check(request: SecurityCheckRequest, db: Session = Depends(get_db)):
    checker = SecurityChecker(db)
    result = checker.check(request.text)
    return SecurityCheckResponse(**result)


@router.get("/prompts", response_model=List[dict])
def get_prompts(category: str = None, db: Session = Depends(get_db)):
    lab = PromptLab(db)
    prompts = lab.get_prompts(category)
    return [{"id": p.id, "name": p.name, "content": p.content, "category": p.category} for p in prompts]


@router.post("/prompts")
def create_prompt(request: PromptRequest, db: Session = Depends(get_db)):
    lab = PromptLab(db)
    prompt = lab.save_prompt(request.name, request.content, request.category)
    return {"id": prompt.id, "name": prompt.name, "content": prompt.content}


@router.post("/prompts/compare")
def compare_prompts(request: ABTestRequest, db: Session = Depends(get_db)):
    lab = PromptLab(db)
    result = lab.ab_test(
        request.prompt_a,
        request.prompt_b,
        request.test_prompt,
        request.model,
        request.system_prompt,
        request.blind_mode
    )
    return result


@router.get("/benchmark/results")
def get_results(model: str = None, limit: int = 50, db: Session = Depends(get_db)):
    generator = ReportGenerator(db)
    results = generator.get_historical_results(model, limit)
    return {"results": results}


@router.post("/benchmark/report")
def generate_report(request: ReportRequest, db: Session = Depends(get_db)):
    generator = ReportGenerator(db)
    report = generator.generate_markdown_report(
        request.model,
        request.benchmark_type,
        request.results
    )
    filepath = generator.save_report(report)
    return {"report": report, "filepath": filepath}


@router.get("/health")
def health_check():
    return {"status": "healthy", "service": "LLM Evaluation Platform"}
