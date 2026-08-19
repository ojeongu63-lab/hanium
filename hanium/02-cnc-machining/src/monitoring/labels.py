import sqlite3
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS qc_labels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id TEXT NOT NULL,
    produced_day INTEGER NOT NULL,
    arrived_day INTEGER NOT NULL,
    label TEXT NOT NULL
)
"""


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(_SCHEMA)
    return conn


def record_label(
    batch_id: str, produced_day: int, arrived_day: int, label: str, db_path: Path
) -> None:
    conn = _connect(db_path)
    with conn:
        conn.execute(
            "INSERT INTO qc_labels (batch_id, produced_day, arrived_day, label) "
            "VALUES (?, ?, ?, ?)",
            (batch_id, produced_day, arrived_day, label),
        )
    conn.close()


def get_arrived_labels(current_day: int, db_path: Path) -> list[dict]:
    """검사 결과가 이미 도착한 라벨만 돌려준다.

    실제 현장에서 QC 결과는 가공 직후가 아니라 며칠 뒤에 나온다. 재학습이
    항상 불완전한 정보로 결정한다는 사실을 이 지연이 그대로 재현한다.
    """
    if not Path(db_path).exists():
        return []
    conn = _connect(db_path)
    rows = conn.execute(
        "SELECT batch_id, produced_day, arrived_day, label FROM qc_labels "
        "WHERE arrived_day <= ? ORDER BY produced_day, id",
        (current_day,),
    ).fetchall()
    conn.close()
    return [
        {"batch_id": r[0], "produced_day": r[1], "arrived_day": r[2], "label": r[3]}
        for r in rows
    ]
