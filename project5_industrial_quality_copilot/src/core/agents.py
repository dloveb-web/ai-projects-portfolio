"""
Multi-agent collaboration system for quality inspection.

Three agents work in a pipeline:
  1. DetectionAgent — runs YOLO/VLM defect detection
  2. AnalysisAgent — root-cause analysis with RAG context
  3. ReportAgent  — structured quality report generation

MultiAgentSystem orchestrates the full pipeline end-to-end.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from .detector import DefectDetector, _get_detector
from .analyzer import DefectAnalyzer, analyzer as default_analyzer
from .rag_engine import RAGEngine, _get_rag_engine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared types
# ---------------------------------------------------------------------------

class AgentStatus(Enum):
    IDLE = "idle"
    WORKING = "working"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class AgentMessage:
    sender: str
    receiver: str
    content: Any
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Base agent
# ---------------------------------------------------------------------------

class BaseAgent:
    """Minimal agent base with message-passing and status tracking."""

    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.status = AgentStatus.IDLE
        self.message_history: List[AgentMessage] = []

    def send_message(
        self,
        receiver: str,
        content: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentMessage:
        message = AgentMessage(
            sender=self.name,
            receiver=receiver,
            content=content,
            metadata=metadata or {},
        )
        self.message_history.append(message)
        logger.info("%s -> %s: %s", self.name, receiver, type(content).__name__)
        return message

    def receive_message(self, message: AgentMessage) -> None:
        self.message_history.append(message)
        logger.info(
            "%s received from %s: %s",
            self.name, message.sender, type(message.content).__name__,
        )

    def reset(self) -> None:
        self.status = AgentStatus.IDLE
        self.message_history.clear()


# ---------------------------------------------------------------------------
# Detection agent
# ---------------------------------------------------------------------------

class DetectionAgent(BaseAgent):
    """Wraps DefectDetector; handles file-path and base64 inputs."""

    def __init__(self, detector_instance: Optional[DefectDetector] = None):
        super().__init__("DetectionAgent", "detect")
        self.detector = detector_instance or _get_detector()

    def detect_defects(self, image_path: str) -> Dict[str, Any]:
        self.status = AgentStatus.WORKING
        try:
            result = self.detector.detect_from_path(image_path)
            self.status = AgentStatus.COMPLETED
            return {"success": True, **result}
        except Exception as exc:
            self.status = AgentStatus.ERROR
            logger.error("Detection error: %s", exc)
            return {"success": False, "error": str(exc)}

    def detect_from_base64(self, image_base64: str) -> Dict[str, Any]:
        self.status = AgentStatus.WORKING
        try:
            result = self.detector.detect_from_base64(image_base64)
            self.status = AgentStatus.COMPLETED
            return {"success": True, **result}
        except Exception as exc:
            self.status = AgentStatus.ERROR
            return {"success": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Analysis agent
# ---------------------------------------------------------------------------

class AnalysisAgent(BaseAgent):
    """Wraps DefectAnalyzer + RAGEngine for rich defect analysis."""

    def __init__(
        self,
        analyzer_instance: Optional[DefectAnalyzer] = None,
        rag_instance: Optional[RAGEngine] = None,
    ):
        super().__init__("AnalysisAgent", "analyze")
        self.analyzer = analyzer_instance or default_analyzer
        self.rag_engine = rag_instance or _get_rag_engine()

    def analyze_defect(
        self,
        defect_info: Dict[str, Any],
        defect_image: Optional[Any] = None,
        include_history: bool = True,
    ) -> Dict[str, Any]:
        self.status = AgentStatus.WORKING
        try:
            defect_type = defect_info.get("class_name", "unknown")

            similar_cases: List[Dict[str, Any]] = []
            if include_history:
                similar_cases = self.rag_engine.find_similar_cases(
                    defect_description=str(defect_info),
                    defect_type=defect_type,
                    top_k=3,
                )

            context = (
                f"Found {len(similar_cases)} similar historical cases"
                if include_history
                else None
            )

            analysis_result = self.analyzer.analyze_defect(
                defect_image=defect_image,
                defect_info=defect_info,
                context=context,
            )

            analysis_result["similar_cases"] = similar_cases
            self.status = AgentStatus.COMPLETED
            return {
                "success": True,
                "analysis": analysis_result,
                "similar_cases_count": len(similar_cases),
            }
        except Exception as exc:
            self.status = AgentStatus.ERROR
            logger.error("Analysis error: %s", exc)
            return {"success": False, "error": str(exc)}

    def batch_analyze(
        self,
        defects: List[Dict[str, Any]],
        defect_images: Optional[List[Any]] = None,
    ) -> List[Dict[str, Any]]:
        results = []
        for i, defect in enumerate(defects):
            image = defect_images[i] if defect_images and i < len(defect_images) else None
            results.append(self.analyze_defect(defect, image))
        return results


# ---------------------------------------------------------------------------
# Report agent
# ---------------------------------------------------------------------------

class ReportAgent(BaseAgent):
    """Generates structured quality inspection reports."""

    def __init__(self):
        super().__init__("ReportAgent", "report")

    def generate_report(
        self,
        detection_result: Dict[str, Any],
        analysis_results: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.status = AgentStatus.WORKING
        try:
            report: Dict[str, Any] = {
                "title": "Quality Inspection Report",
                "generated_at": datetime.now().isoformat(),
                "metadata": metadata or {},
                "sections": {},
            }

            report["sections"]["summary"] = self._generate_summary(detection_result)
            report["sections"]["defect_details"] = self._generate_defect_details(
                detection_result
            )
            report["sections"]["analysis_results"] = self._generate_analysis_section(
                analysis_results
            )
            report["sections"]["recommendations"] = self._generate_recommendations(
                analysis_results
            )

            self.status = AgentStatus.COMPLETED
            return {
                "success": True,
                "report": report,
                "report_id": f"REP_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            }
        except Exception as exc:
            self.status = AgentStatus.ERROR
            logger.error("Report generation error: %s", exc)
            return {"success": False, "error": str(exc)}

    # -- section helpers -----------------------------------------------------

    @staticmethod
    def _generate_summary(detection_result: Dict[str, Any]) -> Dict[str, Any]:
        defects = detection_result.get("defects", [])
        return {
            "total_defects": len(defects),
            "has_defect": detection_result.get("has_defect", False),
            "detection_time": detection_result.get("processing_time", 0),
            "overall_status": "FAIL" if defects else "PASS",
        }

    @staticmethod
    def _generate_defect_details(
        detection_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        return [
            {
                "type": d.get("class_name", "unknown"),
                "confidence": d.get("confidence", 0),
                "location": d.get("bbox", []),
            }
            for d in detection_result.get("defects", [])
        ]

    @staticmethod
    def _generate_analysis_section(
        analysis_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        return [
            {
                "defect_type": r.get("analysis", {}).get("defect_type", "unknown"),
                "severity": r.get("analysis", {}).get("severity", "unknown"),
                "cause": r.get("analysis", {}).get("cause_analysis", ""),
                "suggestions": r.get("analysis", {}).get("suggestions", []),
            }
            for r in analysis_results
            if r.get("success")
        ]

    @staticmethod
    def _generate_recommendations(
        analysis_results: List[Dict[str, Any]],
    ) -> List[str]:
        recommendations: List[str] = []
        for result in analysis_results:
            if result.get("success") and "analysis" in result:
                suggestions = result["analysis"].get("suggestions", [])
                recommendations.extend(suggestions)
        # Deduplicate while preserving order
        seen: set = set()
        unique: List[str] = []
        for r in recommendations:
            if r not in seen:
                seen.add(r)
                unique.append(r)
        return unique[:5]


# ---------------------------------------------------------------------------
# Multi-agent orchestrator
# ---------------------------------------------------------------------------

class MultiAgentSystem:
    """Orchestrates the full detection → analysis → report pipeline."""

    def __init__(self):
        self.detection_agent = DetectionAgent()
        self.analysis_agent = AnalysisAgent()
        self.report_agent = ReportAgent()

    def process_inspection(
        self,
        image_path: Optional[str] = None,
        image_base64: Optional[str] = None,
        include_report: bool = True,
    ) -> Dict[str, Any]:
        """Run the complete inspection pipeline.

        Args:
            image_path: Path to the image file.
            image_base64: Base64-encoded image (used if image_path is None).
            include_report: Whether to generate a final report.

        Returns:
            dict with success, detection, analysis, and optionally report.
        """
        # -- Step 1: Detection -------------------------------------------------
        if image_path:
            detection_result = self.detection_agent.detect_defects(image_path)
        elif image_base64:
            detection_result = self.detection_agent.detect_from_base64(image_base64)
        else:
            return {"success": False, "error": "No image provided"}

        if not detection_result.get("success"):
            return detection_result

        # -- Early return if no defects ---------------------------------------
        if not detection_result.get("has_defect"):
            return {
                "success": True,
                "status": "PASS",
                "detection": detection_result,
                "message": "No defects detected",
            }

        # -- Step 2: Analysis --------------------------------------------------
        analysis_results = self.analysis_agent.batch_analyze(
            detection_result.get("defects", [])
        )

        result: Dict[str, Any] = {
            "success": True,
            "detection": detection_result,
            "analysis": analysis_results,
        }

        # -- Step 3: Report ----------------------------------------------------
        if include_report:
            report_result = self.report_agent.generate_report(
                detection_result, analysis_results
            )
            result["report"] = report_result

        return result


# Module-level singleton
agent_system = MultiAgentSystem()
