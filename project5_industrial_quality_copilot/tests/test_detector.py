"""
Edge-case and integration tests for the defect detector.
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.detector import DefectDetector


class TestDetectorEdgeCases:
    """Test edge cases for DefectDetector."""

    def test_grayscale_image(self):
        detector = DefectDetector()
        gray_image = np.random.randint(0, 255, (640, 640), dtype=np.uint8)
        result = detector.detect(gray_image)
        assert "has_defect" in result

    def test_rgba_image(self):
        detector = DefectDetector()
        rgba_image = np.random.randint(0, 255, (640, 640, 4), dtype=np.uint8)
        result = detector.detect(rgba_image)
        assert "has_defect" in result

    def test_custom_threshold(self):
        detector = DefectDetector()
        test_image = np.zeros((640, 640, 3), dtype=np.uint8)
        result = detector.detect(test_image, conf_threshold=0.9)
        assert "has_defect" in result

    def test_draw_defect_boxes(self):
        detector = DefectDetector()
        image = np.zeros((640, 640, 3), dtype=np.uint8)
        defects = [
            {"bbox": [100, 100, 200, 200], "class_name": "scratches", "confidence": 0.9}
        ]
        annotated = detector.draw_defect_boxes(image, defects)
        assert annotated.shape == image.shape
        # Annotated should differ from original (boxes were drawn)
        assert not np.array_equal(annotated[100:200, 100:200], image[100:200, 100:200])

    def test_get_defect_regions(self):
        detector = DefectDetector()
        image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        defects = [
            {"bbox": [100, 100, 200, 200], "class_name": "scratches", "confidence": 0.9}
        ]
        regions = detector.get_defect_regions(image, defects)
        assert len(regions) == 1
        assert regions[0].shape == (100, 100, 3)

    def test_multiple_defect_regions(self):
        detector = DefectDetector()
        image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        defects = [
            {"bbox": [50, 50, 150, 150], "class_name": "crazing", "confidence": 0.8},
            {"bbox": [300, 300, 400, 400], "class_name": "inclusion", "confidence": 0.7},
        ]
        regions = detector.get_defect_regions(image, defects)
        assert len(regions) == 2

    def test_image_to_base64(self):
        detector = DefectDetector()
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
        assert isinstance(result["defects"], list)

    def test_detect_from_bytes_invalid(self):
        detector = DefectDetector()
        result = detector.detect_from_bytes(b"not an image")
        assert "has_defect" in result
        assert result["defects"] == []

    def test_detect_from_base64(self):
        detector = DefectDetector()
        b64 = detector.image_to_base64(np.zeros((64, 64, 3), dtype=np.uint8))
        result = detector.detect_from_base64(b64)
        assert "has_defect" in result

    def test_detect_string_path_returns_dict(self):
        """detect() with a file path should also return a dict."""
        detector = DefectDetector()
        import tempfile
        import cv2
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            img = np.zeros((64, 64, 3), dtype=np.uint8)
            cv2.imwrite(tmp.name, img)
            tmp_path = tmp.name
        try:
            result = detector.detect(tmp_path)
            assert "has_defect" in result
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_detect_batch(self):
        detector = DefectDetector()
        import tempfile
        import cv2
        paths = []
        for _ in range(3):
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                cv2.imwrite(tmp.name, np.zeros((64, 64, 3), dtype=np.uint8))
                paths.append(tmp.name)
        try:
            results = detector.detect_batch(paths)
            assert len(results) == 3
            for r in results:
                assert "has_defect" in r
        finally:
            for p in paths:
                Path(p).unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
