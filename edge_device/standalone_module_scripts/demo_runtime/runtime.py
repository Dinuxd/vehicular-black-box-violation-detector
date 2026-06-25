from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = PROJECT_ROOT / "demo"
DEFAULT_VENV = PROJECT_ROOT / "shouting" / "venv2"
DROWSINESS_VENV = PROJECT_ROOT / "camera" / "Drowsiness" / ".venv"


def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


@dataclass(slots=True)
class RuntimePaths:
    run_id: str
    proof_dir: Path
    runtime_dir: Path
    outbox_db: Path
    event_log: Path
    process_log_dir: Path

    @classmethod
    def create(cls, run_id: str | None = None) -> "RuntimePaths":
        run_id = run_id or utc_run_id()
        proof_dir = DEMO_DIR / "proof" / run_id
        runtime_dir = DEMO_DIR / "runtime"
        outbox_dir = runtime_dir / "outbox"
        process_log_dir = proof_dir / "logs"
        for path in (proof_dir, runtime_dir, outbox_dir, process_log_dir):
            path.mkdir(parents=True, exist_ok=True)
        return cls(
            run_id=run_id,
            proof_dir=proof_dir,
            runtime_dir=runtime_dir,
            outbox_db=outbox_dir / "events.sqlite3",
            event_log=proof_dir / "events.jsonl",
            process_log_dir=process_log_dir,
        )

