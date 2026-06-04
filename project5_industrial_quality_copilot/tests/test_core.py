"""
Tests for core modules: detector, analyzer, RAG, agents, report generator, database.
"""

from __future__ import annotations

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.detector import DefectDetector, CLASS_NAMES, CLASS_NAMES_CN
from src.core.analyzer import DefectAnalyzer
from src.core.rag_engine import RAGEngine
from src.core.agents import DetectionAgent, AnalysisAgent, ReportAgent, MultiAgentSystem
from src.core.report_generator import ReportGenerator
from src.database.models import User, Task, Detection, Defect


# ---------------------------------------------------------------------------
# DefectDetector
# ---------------------------------------------------------------------------

class TestDefectDetector:
    def test_initialization(self):
        detector = DefectDetector()
        assert detector is not None
        assert detector.confidence_threshold == 0.5
        assert detector.iou_threshold == 0.45

    def test_custom_thresholds(self):
        detector = DefectDetector(confidence_threshold=0.8, iou_threshold=0.3)
        assert detector.confidence_threshold == 0.8
        assert detector.iou_threshold == 0.3

    def test_detect_numpy_array(self):
        detector = DefectDetector()
        test_image = np.zeros((640, 640, 3), dtype=np.uint8)
        result = detector.detect(test_image)
        assert "has_defect" in result
        assert "defects" in result
        assert isinstance(result["defects"], list)

    def test_detect_with_conf_threshold(self):
        detector = DefectDetector()
        test_image = np.zeros((640, 640, 3), dtype=np.uint8)
        result = detector.detect(test_image, conf_threshold=0.9)
        assert "has_defect" in result

    def test_detect_grayscale(self):
        detector = DefectDetector()
        gray_image = np.random.randint(0, 255, (640, 640), dtype=np.uint8)
        result = detector.detect(gray_image)
        assert "has_defect" in result

    def test_detect_rgba(self):
        detector = DefectDetector()
        rgba_image = np.random.randint(0, 255, (640, 640, 4), dtype=np.uint8)
        result = detector.detect(rgba_image)
        assert "has_defect" in result

    def test_draw_defect_boxes(self):
        detector = DefectDetector()
        image = np.zeros((640, 640, 3), dtype=np.uint8)
        defects = [
            {"bbox": [100, 100, 200, 200], "class_name": "scratches", "confidence": 0.9}
        ]
        annotated = detector.draw_defect_boxes(image, defects)
        assert annotated.shape == image.shape

    def test_get_defect_regions(self):
        detector = DefectDetector()
        image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        defects = [
            {"bbox": [100, 100, 200, 200], "class_name": "scratches", "confidence": 0.9}
        ]
        regions = detector.get_defect_regions(image, defects)
        assert len(regions) == 1
        assert regions[0].shape == (100, 100, 3)

    def test_image_to_base64(self):
        detector = DefectDetector()
        import base64
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        b64 = detector.image_to_base64(image)
        assert isinstance(b64, str)
        decoded = base64.b64decode(b64)
        assert len(decoded) > 0

    def test_detect_from_bytes(self):
        detector = DefectDetector()
        import cv2
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", image)
        result = detector.detect_from_bytes(buf.tobytes())
        assert "has_defect" in result

    def test_class_mappings(self):
        assert CLASS_NAMES[0] == "crazing"
        assert CLASS_NAMES_CN[0] == "龟裂"
        assert len(CLASS_NAMES) == 6


# ---------------------------------------------------------------------------
# DefectAnalyzer
# ---------------------------------------------------------------------------

class TestDefectAnalyzer:
    def test_initialization(self):
        analyzer = DefectAnalyzer()
        assert analyzer is not None

    def test_analyze_with_none_image(self):
        analyzer = DefectAnalyzer()
        result = analyzer.analyze_defect(
            defect_image=None,
            defect_info={"class_name": "scratches", "confidence": 0.85},
        )
        assert result is not None
        assert "severity" in result
        assert "cause_analysis" in result
        assert "suggestions" in result

    def test_analyze_known_defect_types(self):
        """All six defect classes should return valid analysis."""
        analyzer = DefectAnalyzer()
        for defect_type in ["crazing", "inclusion", "pitted_surface",
                            "scratches", "patches", "rolled-in_scale"]:
            result = analyzer.analyze_defect(
                defect_image=None,
                defect_info={"class_name": defect_type, "confidence": 0.7},
            )
            assert result["defect_type"] == defect_type
            assert result["severity"] in ("critical", "high", "medium", "low")
            assert len(result["suggestions"]) >= 2

    def test_severity_classification(self):
        assert DefectAnalyzer._classify_severity(0.95) == "critical"
        assert DefectAnalyzer._classify_severity(0.80) == "high"
        assert DefectAnalyzer._classify_severity(0.60) == "medium"
        assert DefectAnalyzer._classify_severity(0.30) == "low"

    def test_batch_analyze(self):
        analyzer = DefectAnalyzer()
        results = analyzer.analyze_batch(
            defect_images=[None, None],
            defect_infos=[
                {"class_name": "scratches", "confidence": 0.8},
                {"class_name": "patches", "confidence": 0.3},
            ],
        )
        assert len(results) == 2
        assert all("severity" in r for r in results)


# ---------------------------------------------------------------------------
# RAGEngine
# ---------------------------------------------------------------------------

class TestRAGEngine:
    def test_initialization(self):
        rag = RAGEngine()
        assert rag is not None
        assert rag.embedding_model_name is not None

    def test_add_document(self):
        rag = RAGEngine()
        success = rag.add_document(
            "test_doc_1",
            "Steel surface scratch caused by conveyor belt misalignment",
            {"category": "defect_case", "defect_type": "scratches"},
        )
        assert isinstance(success, bool)

    def test_search_returns_list(self):
        rag = RAGEngine()
        results = rag.search("scratch defect", top_k=3)
        assert isinstance(results, list)

    def test_find_similar_cases(self):
        rag = RAGEngine()
        cases = rag.find_similar_cases(
            defect_description="scratch on surface",
            defect_type="scratches",
            top_k=3,
        )
        assert isinstance(cases, list)

    def test_search_by_defect_type(self):
        rag = RAGEngine()
        results = rag.search_by_defect_type("scratches", top_k=3)
        assert isinstance(results, list)

    def test_get_collection_count(self):
        rag = RAGEngine()
        count = rag.get_collection_count()
        assert isinstance(count, int)


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

class TestAgents:
    def test_detection_agent(self):
        agent = DetectionAgent()
        assert agent.name == "DetectionAgent"
        assert agent.role == "detect"
        assert agent.status.value == "idle"

    def test_analysis_agent(self):
        agent = AnalysisAgent()
        assert agent.name == "AnalysisAgent"
        assert agent.role == "analyze"

    def test_report_agent(self):
        agent = ReportAgent()
        assert agent.name == "ReportAgent"
        assert agent.role == "report"

    def test_multi_agent_system(self):
        system = MultiAgentSystem()
        assert system.detection_agent is not None
        assert system.analysis_agent is not None
        assert system.report_agent is not None

    def test_multi_agent_no_image(self):
        system = MultiAgentSystem()
        result = system.process_inspection()  # no image provided
        assert result.get("success") is False
        assert "error" in result

    def test_agent_reset(self):
        agent = DetectionAgent()
        agent.status = __import__('src.core.agents', fromlist=['AgentStatus']).AgentStatus.WORKING
        agent.reset()
        assert agent.status.value == "idle"


# ---------------------------------------------------------------------------
# ReportGenerator
# ---------------------------------------------------------------------------

class TestReportGenerator:
    def test_initialization(self):
        gen = ReportGenerator()
        assert gen is not None

    def test_generate_html_pass(self):
        gen = ReportGenerator()
        detection = {"has_defect": False, "defects": [], "processing_time": 0.1}
        html = gen.generate_html_report(detection, [])
        assert "<html" in html
        assert "PASS" in html

    def test_generate_html_fail(self):
        gen = ReportGenerator()
        detection = {
            "has_defect": True,
            "defects": [
                {"class_name": "scratches", "confidence": 0.9, "bbox": [100, 100, 200, 200]}
            ],
            "processing_time": 0.2,
        }
        html = gen.generate_html_report(detection, [])
        assert "<html" in html
        assert "FAIL" in html

    def test_generate_and_save(self):
        gen = ReportGenerator()
        detection = {"has_defect": False, "defects": [], "processing_time": 0.05}
        result = gen.generate_and_save(detection, [])
        assert result["success"] is True
        assert result["report_id"].startswith("QIR_")
        assert Path(result["file_path"]).exists()

    def test_statistics_report(self):
        gen = ReportGenerator()
        detections = [
            {"has_defect": True, "defects": [{"class_name": "scratches"}]},
            {"has_defect": False, "defects": []},
        ]
        stats = gen.generate_statistics_report(detections)
        assert stats["summary"]["total_inspections"] == 2
        assert stats["summary"]["pass_rate"] == 50.0


# ---------------------------------------------------------------------------
# Database models
# ---------------------------------------------------------------------------

class TestDatabaseModels:
    def test_user_model(self):
        user = User(username="test_user", role="engineer")
        assert user.username == "test_user"
        assert user.role == "engineer"

    def test_task_model(self):
        task = Task(status="pending")
        assert task.status == "pending"

    def test_detection_model(self):
        detection = Detection(has_defect=False)
        assert not detection.has_defect

    def test_defect_model(self):
        defect = Defect(
            class_name="scratches",
            confidence=0.95,
            bbox=[100, 100, 200, 200],
        )
        assert defect.class_name == "scratches"
        assert defect.confidence == 0.95
        assert defect.bbox == [100, 100, 200, 200]


@pytest.mark.asyncio
async def test_database_initialization():
    from src.database.connection import init_db
    await init_db()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
