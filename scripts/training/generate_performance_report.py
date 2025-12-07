"""
Generate research-style charts and a concise markdown performance summary.

Outputs (default):
- charts/best_per_model.png
- charts/accuracy_trend.png
- charts/training_curve.png
- training_logs/performance_report.md

Usage:
    python3 scripts/training/generate_performance_report.py \
        --history training_logs/run_history.csv \
        --logs_dir training_logs \
        --out_md training_logs/performance_report.md \
        --charts_dir charts
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


PALETTE = {
    "bg": "#f8fafc",
    "text": "#0f172a",
    "muted": "#94a3b8",
    "blue": "#2563eb",
    "green": "#22c55e",
    "amber": "#f59e0b",
    "red": "#ef4444",
}


def set_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": PALETTE["bg"],
        "axes.edgecolor": PALETTE["muted"],
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.frameon": False,
    })


def load_history(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"No run history at {csv_path}")
    with csv_path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def best_per_model(runs: List[Dict[str, str]]) -> List[Tuple[str, float]]:
    best: Dict[str, float] = {}
    for r in runs:
        model = r.get("model", "")
        try:
            acc = float(r.get("best_acc", "") or 0)
        except ValueError:
            continue
        if not model:
            continue
        best[model] = max(best.get(model, 0.0), acc)
    return sorted(best.items(), key=lambda x: x[1], reverse=True)


def select_best_run(runs: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
    scored = []
    for r in runs:
        try:
            scored.append((float(r.get("best_acc", "") or 0), r))
        except ValueError:
            continue
    if not scored:
        return None
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def best_runs_per_model(runs: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    """Return the best run per model keyed by model name."""
    best: Dict[str, Tuple[float, Dict[str, str]]] = {}
    for r in runs:
        model = r.get("model", "")
        if not model:
            continue
        try:
            acc = float(r.get("best_acc", "") or 0)
        except ValueError:
            continue
        current = best.get(model)
        if current is None or acc > current[0]:
            best[model] = (acc, r)
    return {m: info[1] for m, info in best.items()}

def model_stats(runs: List[Dict[str, str]]) -> List[Tuple[str, float, float, float, float]]:
    """Return (model, mean, std, min, max) for accuracy."""
    per_model: Dict[str, List[float]] = {}
    for r in runs:
        m = r.get("model", "")
        try:
            acc = float(r.get("best_acc", "") or 0)
        except ValueError:
            continue
        if not m:
            continue
        per_model.setdefault(m, []).append(acc)
    stats = []
    for m, vals in per_model.items():
        mean_v = sum(vals) / len(vals)
        std_v = (sum((v - mean_v) ** 2 for v in vals) / max(1, len(vals) - 1)) ** 0.5
        stats.append((m, mean_v, std_v, min(vals), max(vals)))
    stats.sort(key=lambda x: x[1], reverse=True)
    return stats


def plot_bar_with_error(stats: List[Tuple[str, float, float, float, float]], out: Path) -> None:
    if not stats:
        return
    labels = [s[0] for s in stats]
    means = [s[1] for s in stats]
    stds = [s[2] for s in stats]
    plt.figure(figsize=(8, 4.2))
    bars = plt.bar(labels, means, yerr=stds, color=PALETTE["blue"], alpha=0.9, capsize=6, ecolor=PALETTE["muted"])
    plt.ylabel("Validation accuracy (%)")
    plt.title("Mean accuracy ± std per model")
    plt.xticks(rotation=15)
    for bar, mu in zip(bars, means):
        plt.text(bar.get_x() + bar.get_width() / 2, mu + 0.3, f"{mu:.2f}%", ha="center", va="bottom", fontsize=9)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close()


def plot_per_model_box(stats_data: Dict[str, List[float]], out: Path) -> None:
    if not stats_data:
        return
    labels, data = zip(*sorted(stats_data.items(), key=lambda x: sum(x[1]) / len(x[1]), reverse=True))
    plt.figure(figsize=(9, 4.5))
    plt.boxplot(data, tick_labels=labels, patch_artist=True,
                boxprops=dict(facecolor=PALETTE["bg"], color=PALETTE["text"]),
                medianprops=dict(color=PALETTE["blue"], linewidth=2),
                whiskerprops=dict(color=PALETTE["muted"]),
                capprops=dict(color=PALETTE["muted"]))
    plt.xticks(rotation=20)
    plt.ylabel("Validation accuracy (%)")
    plt.title("Accuracy distribution per model")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close()


def parse_training_curve(log_path: Path) -> Dict[str, List[float]]:
    import re

    if not log_path.exists():
        raise FileNotFoundError(log_path)
    pat = re.compile(
        r"Epoch\s+(\d+)/\d+\s+-\s+train_loss:\s+([0-9.]+)\s+-\s+val_loss:\s+([0-9.]+)\s+-\s+val_acc:\s+([0-9.]+)%"
    )
    data = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            m = pat.search(line)
            if m:
                data.append((int(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))))
    if not data:
        raise ValueError(f"No epoch metrics found in {log_path}")
    data.sort(key=lambda x: x[0])
    epochs, train_loss, val_loss, val_acc = zip(*data)
    return {"epochs": epochs, "train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc}


def plot_training_curve(curve: Dict[str, List[float]], out: Path, title: str) -> None:
    plt.figure(figsize=(9, 4.5))
    epochs = curve["epochs"]
    plt.subplot(1, 2, 1)
    plt.plot(epochs, curve["train_loss"], marker="o", color=PALETTE["blue"], linewidth=2, label="Train loss")
    plt.plot(epochs, curve["val_loss"], marker="s", color=PALETTE["amber"], linewidth=2, label="Val loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss vs epoch"); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, curve["val_acc"], marker="^", color=PALETTE["green"], linewidth=2, label="Val accuracy")
    best_idx = max(range(len(curve["val_acc"])), key=lambda i: curve["val_acc"][i])
    best_ep, best_acc = epochs[best_idx], curve["val_acc"][best_idx]
    plt.axhline(best_acc, color=PALETTE["muted"], linestyle="--", linewidth=1)
    plt.axvline(best_ep, color=PALETTE["muted"], linestyle="--", linewidth=1)
    plt.text(best_ep, best_acc + 0.3, f"Best {best_acc:.2f}% @ {best_ep}", ha="center", fontsize=9, color=PALETTE["text"])
    plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)"); plt.title("Accuracy vs epoch"); plt.legend()

    plt.suptitle(title, fontsize=15, y=1.02)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=240, bbox_inches="tight")
    plt.close()


def write_report(md_path: Path,
                 best: List[Tuple[str, float]],
                 best_run: Optional[Dict[str, str]],
                 chart_paths: Dict[str, Path],
                 training_charts: List[Path]) -> None:
    lines = ["# Model Performance Report", ""]
    if best_run:
        lines.append(f"- Best run: **{best_run.get('model','')}** ({best_run.get('timestamp','')}) "
                     f"at **{float(best_run.get('best_acc',0) or 0):.2f}%**")
        lines.append(f"- Epochs: {best_run.get('epochs','?')} | Batch: {best_run.get('batch_size','?')} | LR: {best_run.get('lr','?')}")
        lines.append("")
    if chart_paths.get("bar"):
        lines.append("## Accuracy summary")
        lines.append(f"![Mean/std per model]({chart_paths['bar'].as_posix()})")
    if chart_paths.get("box"):
        lines.append(f"![Accuracy distribution per model]({chart_paths['box'].as_posix()})")
    if training_charts:
        lines.append("## Training curves (best per model)")
        for p in training_charts:
            lines.append(f"![Training curve]({p.as_posix()})")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[INFO] Wrote report to {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate professional charts and a concise performance report.")
    parser.add_argument("--history", type=Path, default=Path("training_logs/run_history.csv"))
    parser.add_argument("--logs_dir", type=Path, default=Path("training_logs"),
                        help="Directory containing per-run logs for learning curves.")
    parser.add_argument("--learning_log", type=Path,
                        help="Specific log file for the training curve; defaults to best run log in logs_dir.")
    parser.add_argument("--charts_dir", type=Path, default=Path("charts"))
    parser.add_argument("--out_md", type=Path, default=Path("training_logs/performance_report.md"))
    args = parser.parse_args()

    set_style()
    runs = load_history(args.history)
    best_run = select_best_run(runs)
    stats = model_stats(runs)
    per_model_data: Dict[str, List[float]] = {}
    for r in runs:
        m = r.get("model", "")
        try:
            acc = float(r.get("best_acc", "") or 0)
        except ValueError:
            continue
        if m:
            per_model_data.setdefault(m, []).append(acc)

    chart_paths = {
        "bar": args.charts_dir / "accuracy_mean_std.png",
        "box": args.charts_dir / "accuracy_box.png",
    }
    plot_bar_with_error(stats, chart_paths["bar"])
    plot_per_model_box(per_model_data, chart_paths["box"])

    training_curves: List[Path] = []
    # Prefer explicit log if provided
    if args.learning_log and args.learning_log.exists():
        try:
            curve = parse_training_curve(args.learning_log)
            out_path = args.charts_dir / f"training_curve_{args.learning_log.stem}.png"
            plot_training_curve(curve, out_path, title=f"Training curve: {args.learning_log.stem}")
            training_curves.append(out_path)
        except Exception as exc:  # pragma: no cover
            print(f"[WARN] Could not render training curve from {args.learning_log}: {exc}")
    else:
        best_per_model_runs = best_runs_per_model(runs)
        for model, run in best_per_model_runs.items():
            log_path = args.logs_dir / f"{run.get('model','')}_{run.get('timestamp','')}.log"
            if not log_path.exists():
                print(f"[WARN] Missing log for {model}: {log_path}")
                continue
            try:
                curve = parse_training_curve(log_path)
                out_path = args.charts_dir / f"training_curve_{model}.png"
                plot_training_curve(curve, out_path, title=f"Training curve: {model}")
                training_curves.append(out_path)
            except Exception as exc:  # pragma: no cover
                print(f"[WARN] Could not render training curve from {log_path}: {exc}")

    write_report(args.out_md, best_per_model(runs), best_run, chart_paths, training_curves)


if __name__ == "__main__":
    main()
