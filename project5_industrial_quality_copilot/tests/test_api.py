import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.main import app

client = TestClient(app)


class TestHealthEndpoints:
    def test_root(self):
        response = client.get("/")
        assert response.status_code == 200
        assert "name" in response.json()

    def test_health(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


class TestAuthEndpoints:
    def test_login(self):
        response = client.post(
            "/api/v1/auth/login",
            json={"username": "test_user", "password": "test_password"}
        )
        assert response.status_code == 200
        assert "access_token" in response.json()


class TestDetectionEndpoints:
    def test_detection_without_image(self):
        response = client.post("/api/v1/detection/detect")
        assert response.status_code == 422


class TestKnowledgeEndpoints:
    def test_search_without_query(self):
        response = client.get("/api/v1/knowledge/search")
        assert response.status_code == 200

    def test_search_with_query(self):
        response = client.get("/api/v1/knowledge/search?query=test")
        assert response.status_code == 200
        assert "results" in response.json()


class TestStatsEndpoints:
    def test_dashboard_stats(self):
        response = client.get("/api/v1/stats/dashboard")
        assert response.status_code == 200
        data = response.json()
        assert "today_inspections" in data
        assert "pass_rate" in data

    def test_defect_trend(self):
        response = client.get("/api/v1/stats/defect-trend?days=7")
        assert response.status_code == 200
        assert "trend" in response.json()


class TestROIEndpoints:
    def test_roi_analysis(self):
        response = client.get("/api/v1/roi/analysis")
        assert response.status_code == 200
        data = response.json()
        assert "summary" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
