"""
Reset training logs for a fresh set of models.

This removes any existing `training_logs/` directory and recreates an empty one
with a `run_history.csv` header compatible with the current RunLogger fields.

Usage:
    python3 scripts/training/reset_training_logs.py
"""
from __future__ import annotations

import shutil
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    log_dir = root / "training_logs"
    if log_dir.exists():
        shutil.rmtree(log_dir)
        print(f"[INFO] Removed existing {log_dir}")
    log_dir.mkdir(parents=True, exist_ok=True)
    header = [
        "timestamp",
        "model",
        "best_epoch",
        "best_acc",
        "best_loss",
        "epochs",
        "batch_size",
        "lr",
        "weight_decay",
        "scheduler",
        "notes",
    ]
    (log_dir / "run_history.csv").write_text(",".join(header) + "\n", encoding="utf-8")
    print(f"[INFO] Initialized {log_dir}/run_history.csv with header.")


if __name__ == "__main__":
    main()
