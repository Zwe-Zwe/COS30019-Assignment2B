from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List


@dataclass
class RunLogger:
    """Helper that mirrors console output to a log file and keeps a summary CSV."""

    model_name: str
    log_dir: Path

    def __post_init__(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_path = self.log_dir / f"{self.model_name}_{self.timestamp}.log"
        self.summary_path = self.log_dir / "run_history.csv"
        self._summary_fields = [
            "timestamp",
            "model",
            "best_epoch",
            "best_acc",
            "best_loss",
            "epochs",
            "batch_size",
            "lr",
            "notes",
        ]
        # Create file so tailing tools can watch it immediately.
        self.log_path.touch()

    def log(self, message: Any) -> None:
        """Prints to console and appends the same text to the log file."""
        text = str(message)
        print(text)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(text + "\n")

    def record_summary(self, summary: Dict[str, Any]) -> None:
        """Appends a single-row summary for this run into run_history.csv."""
        row = {field: "" for field in self._summary_fields}
        row["timestamp"] = self.timestamp
        row["model"] = self.model_name
        for key, value in summary.items():
            if key in row:
                row[key] = value

        write_header = not self.summary_path.exists()
        with self.summary_path.open("a", encoding="utf-8") as f:
            if write_header:
                f.write(",".join(self._summary_fields) + "\n")
            f.write(",".join(str(row[field]) for field in self._summary_fields) + "\n")

