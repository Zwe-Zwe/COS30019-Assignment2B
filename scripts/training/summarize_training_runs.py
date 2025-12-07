import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_runs(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(f"No run history found at {csv_path}")
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        runs = [row for row in reader]
    return runs


def format_markdown_table(runs):
    headers = ["timestamp", "model", "best_acc", "epochs", "batch_size", "lr", "notes"]
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in runs:
        lines.append("| " + " | ".join(r.get(h, "") for h in headers) + " |")
    return "\n".join(lines)


def model_best_acc(runs: List[dict]) -> List[Tuple[str, float]]:
    """
    Aggregate the best recorded accuracy per model across all runs.
    Returns a list sorted by accuracy (desc).
    """
    best: Dict[str, float] = {}
    for r in runs:
        model = r.get("model", "").strip()
        try:
            acc = float(r.get("best_acc", "") or 0)
        except ValueError:
            continue
        if not model:
            continue
        current = best.get(model, 0.0)
        if acc > current:
            best[model] = acc
    return sorted(best.items(), key=lambda x: x[1], reverse=True)


def _import_matplotlib():
    # Lazy import to avoid hard dependency when not plotting.
    import matplotlib.pyplot as plt
    return plt


def save_comparison_chart(best_scores: List[Tuple[str, float]], output_path: Path) -> None:
    """
    Render a simple bar chart comparing best validation accuracy per model.
    """
    if not best_scores:
        raise ValueError("No scores available to plot")

    plt = _import_matplotlib()

    models = [m for m, _ in best_scores]
    accs = [a for _, a in best_scores]

    plt.figure(figsize=(8, 4.5))
    bars = plt.barh(models, accs, color="#38bdf8")
    plt.xlabel("Best validation accuracy (%)")
    plt.title("Model comparison (best runs)")
    plt.xlim(0, max(accs) * 1.1)
    plt.gca().invert_yaxis()  # Highest at top

    for bar, acc in zip(bars, accs):
        plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 f"{acc:.2f}%", va="center", ha="left", fontsize=9, color="#0ea5e9")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_runs_over_time_chart(runs: List[dict], output_path: Path) -> None:
    """
    Scatter plot of all runs ordered by timestamp, coloured by model.
    """
    plt = _import_matplotlib()

    sorted_runs = sorted(runs, key=lambda r: r.get("timestamp", ""))
    xs = []
    ys = []
    labels = []
    for idx, r in enumerate(sorted_runs):
        try:
            acc = float(r.get("best_acc", "") or 0)
        except ValueError:
            continue
        xs.append(idx)
        ys.append(acc)
        labels.append(r.get("model", "unknown"))

    if not xs:
        raise ValueError("No numeric accuracies available for time chart")

    unique_labels = list(dict.fromkeys(labels))
    palette = plt.get_cmap("tab10")
    color_map = {lbl: palette(i % 10) for i, lbl in enumerate(unique_labels)}
    colors = [color_map[lbl] for lbl in labels]

    plt.figure(figsize=(9, 4.5))
    plt.scatter(xs, ys, c=colors, alpha=0.8)
    plt.xlabel("Run order (by timestamp)")
    plt.ylabel("Validation accuracy (%)")
    plt.title("All runs over time")

    # Compact legend
    handles = [plt.Line2D([0], [0], marker="o", color="w", label=lbl,
                          markerfacecolor=color_map[lbl], markersize=8)
               for lbl in unique_labels]
    plt.legend(handles=handles, title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Summarize training runs from training_logs/run_history.csv"
    )
    parser.add_argument("--history", type=Path,
                        default=Path("training_logs/run_history.csv"))
    parser.add_argument("--sort", choices=["acc", "timestamp"], default="acc",
                        help="Sort order for displaying runs")
    parser.add_argument("--output_md", type=Path,
                        help="Optional path to write markdown table summary")
    parser.add_argument("--chart", type=Path,
                        help="Optional path to save a bar chart of best accuracy per model")
    parser.add_argument("--charts", action="store_true",
                        help="Generate a suite of charts into --chart_dir (best-per-model bar, runs-over-time scatter)")
    parser.add_argument("--chart_dir", type=Path, default=Path("charts"),
                        help="Directory to write generated charts when --charts is used")
    args = parser.parse_args()

    runs = load_runs(args.history)
    if args.sort == "acc":
        runs.sort(key=lambda r: float(r.get("best_acc", 0) or 0), reverse=True)
    else:
        runs.sort(key=lambda r: r.get("timestamp", ""), reverse=True)

    print("Training Runs Summary:")
    for run in runs:
        print(f"- {run['timestamp']} | {run['model']} | "
              f"{run['best_acc']}% | epochs={run['epochs']} | "
              f"batch={run['batch_size']} | lr={run['lr']} | notes={run['notes']}")

    if args.output_md:
        args.output_md.write_text(format_markdown_table(runs), encoding="utf-8")
        print(f"[INFO] Markdown summary written to {args.output_md}")

    if args.chart:
        scores = model_best_acc(runs)
        save_comparison_chart(scores, args.chart)
        print(f"[INFO] Model comparison chart saved to {args.chart}")

    if args.charts:
        args.chart_dir.mkdir(parents=True, exist_ok=True)
        scores = model_best_acc(runs)
        best_chart = args.chart_dir / "best_accuracy_per_model.png"
        save_comparison_chart(scores, best_chart)
        print(f"[INFO] Best-per-model chart saved to {best_chart}")

        runs_chart = args.chart_dir / "runs_over_time.png"
        save_runs_over_time_chart(runs, runs_chart)
        print(f"[INFO] Runs-over-time chart saved to {runs_chart}")


if __name__ == "__main__":
    main()
