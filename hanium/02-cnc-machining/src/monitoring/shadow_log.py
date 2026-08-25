import sqlite3
from datetime import datetime, timezone
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS shadow_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id TEXT NOT NULL,
    champion_label TEXT NOT NULL,
    candidate_label TEXT NOT NULL,
    timestamp TEXT NOT NULL
)
"""


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(_SCHEMA)
    return conn


def record_shadow_prediction(
    batch_id: str, champion_label: str, candidate_label: str, db_path: Path
) -> None:
    conn = _connect(db_path)
    with conn:
        conn.execute(
            "INSERT INTO shadow_predictions "
            "(batch_id, champion_label, candidate_label, timestamp) VALUES (?, ?, ?, ?)",
            (batch_id, champion_label, candidate_label, datetime.now(timezone.utc).isoformat()),
        )
    conn.close()


def get_shadow_predictions(batch_ids: list[str], db_path: Path) -> dict[str, dict]:
    if not Path(db_path).exists() or not batch_ids:
        return {}
    conn = _connect(db_path)
    placeholders = ",".join("?" for _ in batch_ids)
    rows = conn.execute(
        f"SELECT batch_id, champion_label, candidate_label FROM shadow_predictions "
        f"WHERE batch_id IN ({placeholders})",
        batch_ids,
    ).fetchall()
    conn.close()
    return {r[0]: {"champion_label": r[1], "candidate_label": r[2]} for r in rows}
