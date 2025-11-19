import argparse
import csv
from pathlib import Path


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


if __name__ == "__main__":
    main()
