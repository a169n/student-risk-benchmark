from dataclasses import dataclass


@dataclass
class TwinSnapshot:
    student_id: str
    snapshot_time: str
    risk_level: str = "unknown"
    final_grade: float | None = None
