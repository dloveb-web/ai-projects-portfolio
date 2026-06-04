"""
Defect analysis module — Qwen-VL based root-cause analysis.

Provides severity classification, cause analysis, and corrective suggestions.
Falls back to rule-based mock analysis when the model is unavailable.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from ..database.settings import settings

logger = logging.getLogger(__name__)


class DefectAnalyzer:
    """Qwen-VL based defect analyzer with rule-based fallback."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.model_path = model_path or settings.qwen_model_path
        self.device = device or settings.qwen_device
        self.model = None
        self.processor = None
        self._load_model()

    # -- model loading -------------------------------------------------------

    def _load_model(self) -> None:
        try:
            from transformers import AutoModelForVisionSeq2Seq, AutoProcessor
            import torch

            self.processor = AutoProcessor.from_pretrained(self.model_path)
            self.model = AutoModelForVisionSeq2Seq.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto",
            )
            logger.info("Loaded Qwen-VL model from %s", self.model_path)
        except Exception as exc:
            logger.warning("Failed to load Qwen-VL model: %s", exc)
            self.model = None
            self.processor = None

    # -- severity ------------------------------------------------------------

    @staticmethod
    def _classify_severity(confidence: float) -> str:
        if confidence >= 0.9:
            return "critical"
        if confidence >= 0.7:
            return "high"
        if confidence >= 0.5:
            return "medium"
        return "low"

    # -- main analysis entry point -------------------------------------------

    def analyze_defect(
        self,
        defect_image: Optional[np.ndarray],
        defect_info: Dict[str, Any],
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Analyze a single defect.

        Args:
            defect_image: np.ndarray of the defect region, or None.
            defect_info: dict with keys class_name, confidence, bbox.
            context: optional additional context string.

        Returns:
            dict with keys defect_type, severity, cause_analysis, suggestions, confidence.
        """
        # If model is not available, use rule-based mock analysis
        if self.model is None or self.processor is None:
            return self._rule_based_analysis(defect_info)

        # Model is available but image may be None → fallback
        if defect_image is None:
            logger.warning("defect_image is None, falling back to rule-based analysis")
            return self._rule_based_analysis(defect_info)

        # Model + image available → run real inference
        try:
            return self._model_analysis(defect_image, defect_info, context)
        except Exception as exc:
            logger.error("Analysis error: %s", exc)
            return self._rule_based_analysis(defect_info)

    # -- model-based analysis ------------------------------------------------

    def _model_analysis(
        self,
        defect_image: np.ndarray,
        defect_info: Dict[str, Any],
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        import torch

        defect_pil = Image.fromarray(defect_image)

        defect_type = defect_info.get("class_name", "unknown")
        confidence = defect_info.get("confidence", 0.5)

        prompt = (
            f"Analyze this defect in an industrial product:\n"
            f"- Detected defect type: {defect_type}\n"
            f"- Confidence: {confidence:.2f}\n"
            f"{'- Additional context: ' + context if context else ''}\n\n"
            f"Please provide:\n"
            f"1. Detailed description of the defect\n"
            f"2. Severity assessment (low/medium/high/critical)\n"
            f"3. Likely root cause(s)\n"
            f"4. Suggested corrective action(s)\n"
        )

        # Fixed: use 'prompt' not undefined 'query'
        query = f"<img>./defect_image</img>\n{prompt}"

        inputs = self.processor(
            text=query,
            images=[defect_pil],
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
            )

        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        return self._parse_analysis_response(response, defect_info)

    # -- rule-based fallback -------------------------------------------------

    def _rule_based_analysis(self, defect_info: Dict[str, Any]) -> Dict[str, Any]:
        """Rule-based analysis when model is unavailable or image is None.

        Defect type names are aligned with the YOLO class names used in detector.py:
        crazing, inclusion, pitted_surface, scratches, patches, rolled-in_scale.
        """
        defect_type = defect_info.get("class_name", "unknown").lower()
        confidence = defect_info.get("confidence", 0.5)

        severity_map: Dict[str, str] = {
            "crazing": "high",
            "inclusion": "medium",
            "pitted_surface": "medium",
            "scratches": "medium",
            "patches": "low",
            "rolled-in_scale": "medium",
        }

        cause_map: Dict[str, str] = {
            "crazing": "Thermal stress or uneven cooling during rolling process",
            "inclusion": "Foreign material entrapped during casting or solidification",
            "pitted_surface": "Localized corrosion or rolling mill scale imprint",
            "scratches": "Improper handling or contact with abrasive surfaces during conveyance",
            "patches": "Uneven cooling, oil residue, or surface oxidation staining",
            "rolled-in_scale": "Mill scale pressed into surface during hot rolling",
        }

        suggestion_map: Dict[str, str] = {
            "crazing": "Optimize cooling rate uniformity; review rolling temperature profile",
            "inclusion": "Improve molten steel filtration; review refractory lining condition",
            "pitted_surface": "Enhance descaling process; inspect roll surface condition",
            "scratches": "Review material handling procedures; inspect conveyor guides and rollers",
            "patches": "Improve surface cleaning; verify uniform cooling spray distribution",
            "rolled-in_scale": "Increase descaling pressure; optimize rolling pass schedule",
        }

        return {
            "defect_type": defect_info.get("class_name", "unknown"),
            "severity": severity_map.get(defect_type, "medium"),
            "cause_analysis": cause_map.get(
                defect_type, "Unknown cause — requires investigation"
            ),
            "suggestions": [
                suggestion_map.get(
                    defect_type, "Conduct detailed root cause analysis"
                ),
                "Document this defect in quality tracking system",
                "Review related process parameters",
            ],
            "confidence": confidence,
        }

    # -- response parsing ----------------------------------------------------

    def _parse_analysis_response(
        self, response: str, defect_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        severity_keywords = {
            "critical": ["critical", "severe", "major"],
            "high": ["high", "serious", "significant"],
            "medium": ["medium", "moderate"],
            "low": ["low", "minor", "slight"],
        }

        detected_severity = "medium"
        for severity, keywords in severity_keywords.items():
            if any(kw in response.lower() for kw in keywords):
                detected_severity = severity
                break

        # Try to extract root cause section
        cause_analysis = response[:500]
        if "root cause" in response.lower():
            cause_analysis = response.split("root cause", 1)[1]

        return {
            "defect_type": defect_info.get("class_name", "unknown"),
            "severity": detected_severity,
            "cause_analysis": cause_analysis,
            "suggestions": [
                "Review process parameters",
                "Document defect pattern",
                "Consider equipment maintenance",
            ],
            "full_analysis": response,
            "confidence": defect_info.get("confidence", 0.5),
        }

    # -- batch ---------------------------------------------------------------

    def analyze_batch(
        self,
        defect_images: List[Optional[np.ndarray]],
        defect_infos: List[Dict[str, Any]],
        context: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        results = []
        for image, info in zip(defect_images, defect_infos):
            results.append(self.analyze_defect(image, info, context))
        return results


# Module-level singleton
analyzer = DefectAnalyzer()
