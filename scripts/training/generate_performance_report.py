import argparse
import csv
import re
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
PALETTE = {
    "blue": "#0ea5e9",
    "blue_dark": "#0284c7",
    "amber": "#f59e0b",
    "amber_dark": "#c2410c",
    "green": "#22c55e",
    "slate": "#0f172a",
    "slate_light": "#e2e8f0",
}


def apply_house_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#f8fafc",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.frameon": False,
    })


apply_house_style()


def load_runs(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"No run history found at {csv_path}")
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def model_best_acc(runs: List[dict]) -> List[Tuple[str, float]]:
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


def top_runs(runs: List[dict], n: int = 5) -> List[dict]:
    scored = []
    for r in runs:
        try:
            acc = float(r.get("best_acc", "") or 0)
        except ValueError:
            continue
        scored.append((acc, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:n]]


def plot_best_per_model(best_scores: List[Tuple[str, float]], output_path: Path) -> None:
    if not best_scores:
        return
    models = [m for m, _ in best_scores]
    accs = [a for _, a in best_scores]
    plt.figure(figsize=(8, 4.5))
    bars = plt.barh(models, accs, color=PALETTE["blue"])
    plt.xlabel("Best validation accuracy (%)")
    plt.title("Best accuracy per model")
    plt.xlim(0, max(accs) * 1.05)
    plt.gca().invert_yaxis()
    for bar, acc in zip(bars, accs):
        plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 f"{acc:.2f}%", va="center", ha="left", fontsize=9, color=PALETTE["blue_dark"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_runs_over_time(runs: List[dict], output_path: Path) -> None:
    if not runs:
        return
    sorted_runs = sorted(runs, key=lambda r: r.get("timestamp", ""))
    xs = []
    ys = []
    labels = []
    cumulative_best = []
    best_so_far = 0.0
    best_point = (None, None, None)  # x, y, label
    for idx, r in enumerate(sorted_runs):
        try:
            acc = float(r.get("best_acc", "") or 0)
        except ValueError:
            continue
        xs.append(idx)
        ys.append(acc)
        labels.append(r.get("model", "unknown"))
        best_so_far = max(best_so_far, acc)
        cumulative_best.append(best_so_far)
        if best_point[1] is None or acc > best_point[1]:
            best_point = (idx, acc, r.get("model", "unknown"))
    if not xs:
        return
    unique_labels = list(dict.fromkeys(labels))
    palette = plt.get_cmap("tab10")
    color_map = {lbl: palette(i % 10) for i, lbl in enumerate(unique_labels)}
    colors = [color_map[lbl] for lbl in labels]
    plt.figure(figsize=(9, 4.5))
    plt.scatter(xs, ys, c=colors, alpha=0.85, edgecolor="#0f172a", linewidth=0.4)
    plt.plot(xs, cumulative_best, color=PALETTE["green"], linewidth=2, label="Cumulative best")
    if best_point[0] is not None:
        plt.scatter([best_point[0]], [best_point[1]], color=PALETTE["amber"], marker="*", s=160,
                    edgecolor=PALETTE["slate"], linewidth=1.2, zorder=5,
                    label=f"Best: {best_point[2]} ({best_point[1]:.2f}%)")
        plt.annotate(f"{best_point[1]:.2f}%",
                     (best_point[0], best_point[1]),
                     textcoords="offset points", xytext=(6, 6),
                     fontsize=9, color=PALETTE["slate"])
    plt.xlabel("Run order (by timestamp)")
    plt.ylabel("Validation accuracy (%)")
    plt.title("Runs over time")
    handles = [plt.Line2D([0], [0], marker="o", color="w", label=lbl,
                          markerfacecolor=color_map[lbl], markersize=8, markeredgecolor=PALETTE["slate"])
               for lbl in unique_labels]
    handles.append(plt.Line2D([0], [0], color=PALETTE["green"], label="Cumulative best"))
    if best_point[0] is not None:
        handles.append(plt.Line2D([0], [0], marker="*", color=PALETTE["amber"],
                                  markeredgecolor=PALETTE["slate"], markerfacecolor=PALETTE["amber"],
                                  markersize=10, linewidth=0, label="Best run"))
    plt.legend(handles=handles, title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_accuracy_hist(runs: List[dict], output_path: Path) -> None:
    accs = []
    for r in runs:
        try:
            accs.append(float(r.get("best_acc", "") or 0))
        except ValueError:
            continue
    if not accs:
        return
    mean_acc = statistics.mean(accs)
    median_acc = statistics.median(accs)
    plt.figure(figsize=(7.5, 4))
    plt.hist(accs, bins=10, color=PALETTE["green"], edgecolor=PALETTE["slate"], alpha=0.85)
    plt.xlabel("Validation accuracy (%)")
    plt.ylabel("Count")
    plt.title("Accuracy distribution across runs")
    plt.axvline(mean_acc, color=PALETTE["blue"], linestyle="--", linewidth=1.5, label=f"Mean {mean_acc:.1f}%")
    plt.axvline(median_acc, color=PALETTE["amber"], linestyle="-.", linewidth=1.5, label=f"Median {median_acc:.1f}%")
    plt.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_top_runs(top: List[dict], output_path: Path) -> None:
    if not top:
        return
    labels = [f"{r.get('model','')} ({r.get('timestamp','')})" for r in top]
    accs = [float(r.get("best_acc", "") or 0) for r in top]
    plt.figure(figsize=(9, 4.5))
    bars = plt.barh(labels, accs, color=PALETTE["amber"])
    plt.xlabel("Validation accuracy (%)")
    plt.title("Top runs")
    plt.xlim(0, max(accs) * 1.05)
    plt.gca().invert_yaxis()
    for bar, acc in zip(bars, accs):
        plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 f"{acc:.2f}%", va="center", ha="left", fontsize=9, color=PALETTE["amber_dark"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def parse_learning_curve(log_path: Path) -> Dict[str, List[float]]:
    """
    Extract per-epoch train/val loss and val accuracy from a training log.
    """
    if not log_path.exists():
        raise FileNotFoundError(log_path)

    pattern = re.compile(
        r"Epoch\s+(\d+)/\d+\s+-\s+train_loss:\s+([0-9.]+)\s+-\s+val_loss:\s+([0-9.]+)\s+-\s+val_acc:\s+([0-9.]+)%"
    )
    epochs: List[int] = []
    train_loss: List[float] = []
    val_loss: List[float] = []
    val_acc: List[float] = []

    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)
            if not match:
                continue
            epochs.append(int(match.group(1)))
            train_loss.append(float(match.group(2)))
            val_loss.append(float(match.group(3)))
            val_acc.append(float(match.group(4)))

    if not epochs:
        raise ValueError("No epoch rows found in log")

    # Ensure in order (defensive in case of shuffling)
    ordered = sorted(zip(epochs, train_loss, val_loss, val_acc), key=lambda x: x[0])
    epochs, train_loss, val_loss, val_acc = map(list, zip(*ordered))
    return {
        "epochs": epochs,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_acc": val_acc,
    }


def plot_learning_curve(curve: Dict[str, List[float]],
                        output_path: Path,
                        title: str) -> None:
    plt.figure(figsize=(9, 4.5))
    epochs = curve["epochs"]
    best_idx = max(range(len(curve["val_acc"])), key=lambda i: curve["val_acc"][i])
    best_epoch = epochs[best_idx]
    best_acc = curve["val_acc"][best_idx]

    ax1 = plt.gca()
    ax1.plot(epochs, curve["train_loss"], label="Train loss",
             color=PALETTE["blue"], marker="o", linewidth=2)
    ax1.plot(epochs, curve["val_loss"], label="Val loss",
             color=PALETTE["amber"], marker="o", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")

    ax2 = ax1.twinx()
    ax2.plot(epochs, curve["val_acc"], label="Val accuracy (%)",
             color=PALETTE["green"], linestyle="--", linewidth=2, marker="s", markersize=5)
    ax2.set_ylabel("Validation accuracy (%)")
    ax2.axvline(best_epoch, color=PALETTE["slate"], linestyle="--", alpha=0.5)
    ax2.scatter([best_epoch], [best_acc], color=PALETTE["green"], edgecolor=PALETTE["slate"], zorder=5)
    ax2.text(best_epoch, best_acc + 0.5,
             f"Best {best_acc:.2f}% @ epoch {best_epoch}",
             ha="center", va="bottom", fontsize=9, color=PALETTE["slate"])

    lines, labels = [], []
    for ax in (ax1, ax2):
        ln, lb = ax.get_legend_handles_labels()
        lines += ln
        labels += lb
    ax1.legend(lines, labels, loc="center right", bbox_to_anchor=(1.18, 0.5))

    plt.title(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def write_report(md_path: Path,
                 runs: List[dict],
                 best_scores: List[Tuple[str, float]],
                 top: List[dict],
                 chart_paths: Dict[str, Path],
                 best_run: Optional[dict]) -> None:
    lines = []
    lines.append("# Model Performance Report")
    lines.append("")
    if runs:
        lines.append(f"- Total runs: **{len(runs)}**")
        ref = best_run or max(runs, key=lambda r: float(r.get("best_acc", 0) or 0))
        lines.append(f"- Best overall: **{ref.get('model','')}** "
                     f"({ref.get('timestamp','')}) "
                     f"at **{float(ref.get('best_acc',0) or 0):.2f}%**")
        accs = []
        for r in runs:
            try:
                accs.append(float(r.get("best_acc", "") or 0))
            except ValueError:
                continue
        if accs:
            lines.append(f"- Mean accuracy: **{statistics.mean(accs):.2f}%**; "
                         f"median: **{statistics.median(accs):.2f}%**")
    lines.append("")
    if chart_paths.get("best_per_model"):
        lines.append(f"![Best per model]({chart_paths['best_per_model'].as_posix()})")
    if chart_paths.get("top_runs"):
        lines.append(f"![Top runs]({chart_paths['top_runs'].as_posix()})")
    if chart_paths.get("runs_over_time"):
        lines.append(f"![Runs over time]({chart_paths['runs_over_time'].as_posix()})")
    if chart_paths.get("acc_hist"):
        lines.append(f"![Accuracy distribution]({chart_paths['acc_hist'].as_posix()})")
    if chart_paths.get("learning_curve"):
        tag = ""
        if best_run:
            tag = f" ({best_run.get('model','')} {best_run.get('timestamp','')})"
        lines.append(f"![Learning curve{tag}]({chart_paths['learning_curve'].as_posix()})")
    lines.append("")

    def table(headers, rows):
        out = ["| " + " | ".join(headers) + " |",
               "| " + " | ".join(["---"] * len(headers)) + " |"]
        out.extend(["| " + " | ".join(row) + " |" for row in rows])
        return "\n".join(out)

    if best_scores:
        lines.append("## Best accuracy per model")
        rows = [[m, f"{acc:.2f}%"] for m, acc in best_scores]
        lines.append(table(["Model", "Best Acc"], rows))
        lines.append("")
    if top:
        lines.append("## Top runs")
        rows = []
        for r in top:
            rows.append([
                r.get("timestamp", ""),
                r.get("model", ""),
                f"{float(r.get('best_acc',0) or 0):.2f}%",
                r.get("epochs", ""),
                r.get("batch_size", ""),
                r.get("lr", ""),
                r.get("notes", ""),
            ])
        lines.append(table(
            ["Timestamp", "Model", "Best Acc", "Epochs", "Batch", "LR", "Notes"],
            rows))
        lines.append("")

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[INFO] Wrote performance report to {md_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate charts and a markdown report from training_logs/run_history.csv"
    )
    parser.add_argument("--history", type=Path,
                        default=Path("training_logs/run_history.csv"))
    parser.add_argument("--charts_dir", type=Path, default=Path("charts"))
    parser.add_argument("--out_md", type=Path,
                        default=Path("training_logs/performance_report.md"))
    parser.add_argument("--top", type=int, default=5,
                        help="Number of top runs to include")
    parser.add_argument("--logs_dir", type=Path,
                        help="Directory containing per-run logs (defaults to the history file's parent)")
    parser.add_argument("--learning_curve_log", type=Path,
                        help="Specific log file to parse for the learning curve chart; "
                             "defaults to best run's log inside --logs_dir")
    args = parser.parse_args()

    runs = load_runs(args.history)
    best_run = max(runs, key=lambda r: float(r.get("best_acc", 0) or 0)) if runs else None
    best_scores = model_best_acc(runs)
    top = top_runs(runs, n=args.top)

    chart_paths: Dict[str, Path] = {}
    chart_paths["best_per_model"] = args.charts_dir / "best_accuracy_per_model.png"
    chart_paths["runs_over_time"] = args.charts_dir / "runs_over_time.png"
    chart_paths["acc_hist"] = args.charts_dir / "accuracy_distribution.png"
    chart_paths["top_runs"] = args.charts_dir / "top_runs.png"

    plot_best_per_model(best_scores, chart_paths["best_per_model"])
    plot_runs_over_time(runs, chart_paths["runs_over_time"])
    plot_accuracy_hist(runs, chart_paths["acc_hist"])
    plot_top_runs(top, chart_paths["top_runs"])
    logs_dir = args.logs_dir or args.history.parent
    learning_log = args.learning_curve_log
    if not learning_log and best_run:
        learning_log = logs_dir / f"{best_run.get('model','')}_{best_run.get('timestamp','')}.log"
    if learning_log:
        try:
            curve = parse_learning_curve(learning_log)
            chart_paths["learning_curve"] = args.charts_dir / "learning_curve.png"
            title = "Learning curve"
            if best_run:
                title = f"Learning curve: {best_run.get('model','')} ({best_run.get('timestamp','')})"
            plot_learning_curve(curve, chart_paths["learning_curve"], title=title)
            print(f"[INFO] Learning curve chart saved to {chart_paths['learning_curve']} "
                  f"(source: {learning_log})")
        except FileNotFoundError:
            print(f"[WARN] Learning curve log not found: {learning_log}")
        except ValueError as exc:
            print(f"[WARN] Could not parse learning curve from {learning_log}: {exc}")
    write_report(args.out_md, runs, best_scores, top, chart_paths, best_run)


if __name__ == "__main__":
    main()
