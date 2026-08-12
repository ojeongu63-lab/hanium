import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def _init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predict_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            feature_means_json TEXT NOT NULL,
            score REAL NOT NULL,
            predicted_label_text TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def log_request(
    feature_means: dict[str, float],
    score: float,
    predicted_label_text: str,
    db_path: Path,
) -> None:
    _init_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO predict_log (timestamp, feature_means_json, score, predicted_label_text) "
        "VALUES (?, ?, ?, ?)",
        (
            datetime.now(timezone.utc).isoformat(),
            json.dumps(feature_means),
            score,
            predicted_label_text,
        ),
    )
    conn.commit()
    conn.close()


def get_recent_requests(n: int, db_path: Path) -> list[dict]:
    if not db_path.exists():
        return []
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT timestamp, feature_means_json, score, predicted_label_text "
        "FROM predict_log ORDER BY id DESC LIMIT ?",
        (n,),
    ).fetchall()
    conn.close()
    return [
        {
            "timestamp": row["timestamp"],
            "feature_means": json.loads(row["feature_means_json"]),
            "score": row["score"],
            "predicted_label_text": row["predicted_label_text"],
        }
        for row in rows
    ]
