from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import logging

from ..database.settings import get_report_dir

logger = logging.getLogger(__name__)


class ReportGenerator:
    def __init__(self):
        self.report_dir = get_report_dir()

    def generate_html_report(
        self,
        detection_result: Dict[str, Any],
        analysis_results: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        defects = detection_result.get('defects', [])
        total_defects = len(defects)
        has_defect = detection_result.get('has_defect', False)

        defects_html = ""
        for i, defect in enumerate(defects, 1):
            analysis = None
            for a in analysis_results:
                if a.get('success') and a.get('analysis', {}).get('defect_type') == defect.get('class_name'):
                    analysis = a.get('analysis', {})
                    break

            severity_color = self._get_severity_color(analysis.get('severity', 'low') if analysis else 'low')

            defects_html += f"""
            <div class="defect-card">
                <div class="defect-header">
                    <span class="defect-number">Defect #{i}</span>
                    <span class="defect-type">{defect.get('class_name', 'Unknown')}</span>
                    <span class="defect-confidence">Confidence: {defect.get('confidence', 0):.2%}</span>
                </div>
                <div class="defect-body">
                    <div class="severity-indicator" style="background-color: {severity_color}">
                        Severity: {analysis.get('severity', 'N/A') if analysis else 'N/A'}
                    </div>
                    {f'<p class="cause-analysis"><strong>Root Cause:</strong> {analysis.get("cause_analysis", "N/A")}</p>' if analysis and analysis.get('cause_analysis') else ''}
                    {self._generate_suggestions_html(analysis.get('suggestions', []) if analysis else [])}
                </div>
            </div>
            """

        status_class = "pass" if not has_defect else "fail"
        status_text = "PASS" if not has_defect else "FAIL"
        status_color = "#4ecca3" if not has_defect else "#e94560"

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quality Inspection Report - {datetime.now().strftime('%Y-%m-%d')}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #1a1a2e; color: #eaeaea; min-height: 100vh; padding: 40px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ text-align: center; margin-bottom: 40px; }}
        .header h1 {{ color: #4ecca3; font-size: 2.5em; margin-bottom: 10px; }}
        .header .timestamp {{ color: #888; }}
        .status-banner {{ text-align: center; padding: 30px; border-radius: 10px; margin-bottom: 30px; }}
        .status-banner.pass {{ background: linear-gradient(135deg, #16213e, #1a1a2e); border: 2px solid #4ecca3; }}
        .status-banner.fail {{ background: linear-gradient(135deg, #16213e, #1a1a2e); border: 2px solid #e94560; }}
        .status-text {{ font-size: 3em; font-weight: bold; }}
        .status-banner.pass .status-text {{ color: #4ecca3; }}
        .status-banner.fail .status-text {{ color: #e94560; }}
        .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 40px; }}
        .summary-card {{ background: #16213e; padding: 20px; border-radius: 10px; text-align: center; }}
        .summary-card .value {{ font-size: 2.5em; font-weight: bold; color: #4ecca3; }}
        .summary-card .label {{ color: #888; margin-top: 10px; }}
        .defects-section {{ margin-top: 40px; }}
        .defects-section h2 {{ color: #e94560; margin-bottom: 20px; }}
        .defect-card {{ background: #16213e; border-radius: 10px; padding: 20px; margin-bottom: 20px; border-left: 4px solid #e94560; }}
        .defect-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; }}
        .defect-number {{ color: #888; }}
        .defect-type {{ color: #4ecca3; font-weight: bold; font-size: 1.2em; }}
        .defect-confidence {{ color: #888; }}
        .severity-indicator {{ display: inline-block; padding: 5px 15px; border-radius: 5px; color: white; font-weight: bold; margin: 10px 0; }}
        .cause-analysis {{ margin: 15px 0; line-height: 1.6; }}
        .suggestions {{ margin-top: 15px; }}
        .suggestions h4 {{ color: #4ecca3; margin-bottom: 10px; }}
        .suggestions ul {{ list-style: none; }}
        .suggestions li {{ padding: 5px 0; padding-left: 20px; position: relative; }}
        .suggestions li::before {{ content: "→"; position: absolute; left: 0; color: #e94560; }}
        .footer {{ margin-top: 60px; text-align: center; color: #666; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏭 Quality Inspection Report</h1>
            <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>

        <div class="status-banner {status_class}">
            <div class="status-text">{status_text}</div>
            <div>{'No defects detected - Product meets quality standards' if not has_defect else f'{total_defects} defect(s) detected - Product requires review'}</div>
        </div>

        <div class="summary-grid">
            <div class="summary-card">
                <div class="value">{total_defects}</div>
                <div class="label">Total Defects</div>
            </div>
            <div class="summary-card">
                <div class="value">{detection_result.get('processing_time', 0):.3f}s</div>
                <div class="label">Processing Time</div>
            </div>
            <div class="summary-card">
                <div class="value">{len([a for a in analysis_results if a.get('analysis', {}).get('severity') == 'critical'])}</div>
                <div class="label">Critical Issues</div>
            </div>
            <div class="summary-card">
                <div class="value">{datetime.now().strftime('%Y-%m-%d')}</div>
                <div class="label">Inspection Date</div>
            </div>
        </div>

        <div class="defects-section">
            <h2>📋 Defect Details</h2>
            {defects_html if defects_html else '<p>No defects found.</p>'}
        </div>

        <div class="footer">
            <p>Industrial AI Quality Copilot - Automated Inspection System</p>
            <p>Report ID: {metadata.get('report_id', 'N/A') if metadata else 'N/A'}</p>
        </div>
    </div>
</body>
</html>
        """
        return html

    def _get_severity_color(self, severity: str) -> str:
        colors = {
            'critical': '#e94560',
            'high': '#ff6b6b',
            'medium': '#feca57',
            'low': '#4ecca3'
        }
        return colors.get(severity.lower(), '#888')

    def _generate_suggestions_html(self, suggestions: List[str]) -> str:
        if not suggestions:
            return ""
        items = "".join([f"<li>{s}</li>" for s in suggestions[:5]])
        return f"""
        <div class="suggestions">
            <h4>💡 Recommendations:</h4>
            <ul>{items}</ul>
        </div>
        """

    def save_report(
        self,
        report_content: str,
        report_id: str,
        format: str = "html"
    ) -> str:
        if format == "html":
            file_path = self.report_dir / f"{report_id}.html"
        else:
            file_path = self.report_dir / f"{report_id}.json"

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        return str(file_path)

    def generate_and_save(
        self,
        detection_result: Dict[str, Any],
        analysis_results: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        report_id = f"QIR_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        html_report = self.generate_html_report(detection_result, analysis_results, metadata)
        file_path = self.save_report(html_report, report_id, "html")

        return {
            'success': True,
            'report_id': report_id,
            'file_path': file_path,
            'generated_at': datetime.now().isoformat()
        }

    def generate_statistics_report(
        self,
        detections: List[Dict[str, Any]],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        total_inspections = len(detections)
        total_defects = sum(1 for d in detections if d.get('has_defect', False))
        pass_rate = (total_inspections - total_defects) / total_inspections * 100 if total_inspections > 0 else 100

        defect_types = {}
        severity_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}

        for detection in detections:
            for defect in detection.get('defects', []):
                defect_type = defect.get('class_name', 'unknown')
                defect_types[defect_type] = defect_types.get(defect_type, 0) + 1

        return {
            'period': {
                'start': start_date.isoformat() if start_date else None,
                'end': end_date.isoformat() if end_date else None
            },
            'summary': {
                'total_inspections': total_inspections,
                'total_defects': total_defects,
                'pass_rate': pass_rate,
                'defect_rate': 100 - pass_rate
            },
            'defect_types': defect_types,
            'severity_distribution': severity_counts,
            'generated_at': datetime.now().isoformat()
        }


report_generator = ReportGenerator()
