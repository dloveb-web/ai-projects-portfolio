"""
Synchronous inspection database (SQLite).

Used by the Gradio UI for logging, statistics, and history queries.
Shares the SAME database file as the async SQLAlchemy layer (data/quality_copilot.db)
so that data written by the Gradio UI is visible to the FastAPI dashboard and vice-versa.

The async layer manages: users, tasks, detections, defects, analyses, reports, knowledge_base
This module adds:       detection_log (detailed log), training_log (training runs)
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .settings import settings

logger = logging.getLogger(__name__)

# Extract the file path from the async DB URL (e.g. "sqlite+aiosqlite:///./data/quality_copilot.db")
_DB_URL = settings.database_url.replace("sqlite+aiosqlite:///", "").replace("sqlite:///", "")
DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / _DB_URL


class InspectionDB:
    """Synchronous SQLite inspection database, sharing the async DB file."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _get_conn(self):
        import sqlite3
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self):
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS detection_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_path TEXT NOT NULL,
                    image_name TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    num_detections INTEGER DEFAULT 0,
                    detections_json TEXT,
                    vlm_text TEXT DEFAULT '',
                    vlm_defect_types TEXT DEFAULT '',
                    inference_time REAL DEFAULT 0,
                    review_status TEXT DEFAULT 'pending',
                    review_note TEXT DEFAULT '',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_dl_review_status ON detection_log(review_status);
                CREATE INDEX IF NOT EXISTS idx_dl_model_type ON detection_log(model_type);
                CREATE INDEX IF NOT EXISTS idx_dl_created_at ON detection_log(created_at);

                CREATE TABLE IF NOT EXISTS training_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy TEXT NOT NULL,
                    epochs INTEGER,
                    batch_size INTEGER,
                    model_path TEXT,
                    best_map50 REAL,
                    best_map REAL,
                    status TEXT DEFAULT 'running',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_tl_status ON training_log(status);
            """)

    # ------------------------------------------------------------------
    # Detection log
    # ------------------------------------------------------------------

    def save_detection(self, result) -> int:
        """Save a detection result. Accepts both dict and DetectionResult objects."""
        if hasattr(result, 'to_dict'):
            data = result.to_dict()
        else:
            data = result

        image_name = os.path.basename(data.get("image_path", ""))
        boxes = data.get("boxes", [])
        if hasattr(boxes, '__iter__') and not isinstance(boxes, (list, str)):
            boxes = [b.to_dict() if hasattr(b, 'to_dict') else b for b in boxes]

        with self._get_conn() as conn:
            conn.execute("""
                INSERT INTO detection_log
                    (image_path, image_name, model_type, num_detections,
                     detections_json, vlm_text, vlm_defect_types, inference_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data.get("image_path", ""),
                image_name,
                data.get("model_type", "unknown"),
                len(boxes),
                json.dumps(boxes, ensure_ascii=False),
                data.get("vlm_text", ""),
                json.dumps(data.get("vlm_defect_types", []), ensure_ascii=False),
                data.get("inference_time", 0),
            ))

            log_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            return log_id

    def update_review(self, log_id: int, status: str, note: str = "") -> bool:
        with self._get_conn() as conn:
            conn.execute("""
                UPDATE detection_log
                SET review_status = ?, review_note = ?
                WHERE id = ?
            """, (status, note, log_id))
            return True

    def get_records(
        self,
        model_type: Optional[str] = None,
        review_status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        conditions = []
        params: list = []

        if model_type and model_type != "all":
            conditions.append("model_type = ?")
            params.append(model_type)
        if review_status and review_status != "all":
            conditions.append("review_status = ?")
            params.append(review_status)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        query = f"""
            SELECT * FROM detection_log
            {where}
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])

        with self._get_conn() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def get_record_by_id(self, log_id: int) -> Optional[Dict[str, Any]]:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM detection_log WHERE id = ?", (log_id,)
            ).fetchone()
            return dict(row) if row else None

    def get_statistics(self) -> Dict[str, Any]:
        with self._get_conn() as conn:
            stats: Dict[str, Any] = {}

            row = conn.execute("SELECT COUNT(*) as total FROM detection_log").fetchone()
            stats["total"] = row["total"]

            rows = conn.execute("""
                SELECT model_type, COUNT(*) as count
                FROM detection_log GROUP BY model_type
            """).fetchall()
            stats["by_model"] = {row["model_type"]: row["count"] for row in rows}

            rows = conn.execute("""
                SELECT review_status, COUNT(*) as count
                FROM detection_log GROUP BY review_status
            """).fetchall()
            stats["by_status"] = {row["review_status"]: row["count"] for row in rows}

            rows = conn.execute("""
                SELECT model_type, AVG(inference_time) as avg_time
                FROM detection_log GROUP BY model_type
            """).fetchall()
            stats["avg_inference_time"] = {
                row["model_type"]: round(row["avg_time"], 3) for row in rows
            }

            # Defect class distribution
            rows = conn.execute("""
                SELECT detections_json FROM detection_log
                WHERE model_type = 'yolo' AND detections_json != '[]'
            """).fetchall()

            class_counts: Dict[str, int] = {}
            for row in rows:
                try:
                    boxes = json.loads(row["detections_json"])
                    for box in boxes:
                        cls_name = box.get("class_name", "unknown")
                        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                except (json.JSONDecodeError, TypeError):
                    pass
            stats["defect_class_counts"] = class_counts

            # Review accuracy
            row = conn.execute("""
                SELECT
                    COUNT(CASE WHEN review_status = 'correct' THEN 1 END) as correct,
                    COUNT(CASE WHEN review_status IN ('correct', 'wrong', 'missed') THEN 1 END) as reviewed
                FROM detection_log
            """).fetchone()
            if row["reviewed"] > 0:
                stats["accuracy"] = round(row["correct"] / row["reviewed"] * 100, 1)
            else:
                stats["accuracy"] = None
            stats["reviewed_count"] = row["reviewed"]

            return stats

    def get_bad_cases(self, model_type: Optional[str] = None) -> List[Dict[str, Any]]:
        conditions = ["review_status IN ('wrong', 'missed')"]
        params: list = []

        if model_type and model_type != "all":
            conditions.append("model_type = ?")
            params.append(model_type)

        where = "WHERE " + " AND ".join(conditions)

        with self._get_conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM detection_log {where} ORDER BY created_at DESC",
                params,
            ).fetchall()
            return [dict(row) for row in rows]

    def delete_record(self, log_id: int) -> bool:
        with self._get_conn() as conn:
            conn.execute("DELETE FROM detection_log WHERE id = ?", (log_id,))
            return True

    # ------------------------------------------------------------------
    # Training log
    # ------------------------------------------------------------------

    def save_training_log(
        self,
        strategy: str,
        epochs: int,
        batch_size: int,
        model_path: str = "",
    ) -> int:
        with self._get_conn() as conn:
            conn.execute("""
                INSERT INTO training_log (strategy, epochs, batch_size, model_path)
                VALUES (?, ?, ?, ?)
            """, (strategy, epochs, batch_size, model_path))
            return conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    def update_training_log(
        self,
        log_id: int,
        status: str,
        best_map50: Optional[float] = None,
        best_map: Optional[float] = None,
        model_path: Optional[str] = None,
    ) -> bool:
        completed_at = (
            datetime.now().isoformat() if status in ("completed", "failed") else None
        )

        with self._get_conn() as conn:
            if best_map50 is not None:
                conn.execute(
                    "UPDATE training_log SET best_map50 = ? WHERE id = ?",
                    (best_map50, log_id),
                )
            if best_map is not None:
                conn.execute(
                    "UPDATE training_log SET best_map = ? WHERE id = ?",
                    (best_map, log_id),
                )
            if model_path:
                conn.execute(
                    "UPDATE training_log SET model_path = ? WHERE id = ?",
                    (model_path, log_id),
                )
            conn.execute(
                "UPDATE training_log SET status = ?, completed_at = ? WHERE id = ?",
                (status, completed_at, log_id),
            )
            return True

    def get_training_logs(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            rows = conn.execute("""
                SELECT * FROM training_log
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,)).fetchall()
            return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Cross-table queries (unified view)
    # ------------------------------------------------------------------

    def get_unified_statistics(self) -> Dict[str, Any]:
        """Return statistics from BOTH detection_log AND async tables."""
        stats = self.get_statistics()

        # Also count from the async detections table if it exists
        with self._get_conn() as conn:
            try:
                row = conn.execute(
                    "SELECT COUNT(*) as total FROM detections"
                ).fetchone()
                stats["async_detections_total"] = row["total"]
            except Exception:
                stats["async_detections_total"] = 0

        return stats


# Module-level singleton
inspection_db = InspectionDB()
