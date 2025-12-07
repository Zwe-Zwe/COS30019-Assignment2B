import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ORDER = [
    "transfer_resnet18",
    "transfer_mobilenet_v3_small",
    "efficientnet_b0",
]


def parse_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def add_common_args(cmd: List[str],
                    *,
                    data_root: Path,
                    log_dir: Path,
                    output_dir: Path,
                    epochs: int | None,
                    batch_size: int | None,
                    num_workers: int | None) -> None:
    cmd.extend(["--data_root", str(data_root)])
    cmd.extend(["--log_dir", str(log_dir)])
    cmd.extend(["--output_dir", str(output_dir)])
    if epochs is not None:
        cmd.extend(["--epochs", str(epochs)])
    if batch_size is not None:
        cmd.extend(["--batch_size", str(batch_size)])
    if num_workers is not None:
        cmd.extend(["--num_workers", str(num_workers)])


def build_commands(args) -> List[Tuple[str, List[str]]]:
    cmds: List[Tuple[str, List[str]]] = []

    def want(name: str) -> bool:
        if args.models:
            return name in args.models
        return name in args.include

    # Transfer CNN (ResNet18)
    if want("transfer_resnet18"):
        cmd = [sys.executable, str(ROOT / "scripts/training/train_transfer_cnn.py"),
               "--model", "resnet18"]
        add_common_args(cmd, data_root=args.data_root, log_dir=args.log_dir,
                        output_dir=args.output_dir, epochs=args.epochs,
                        batch_size=args.batch_size, num_workers=args.num_workers)
        if args.amp is True:
            cmd.append("--amp")
        elif args.amp is False:
            cmd.append("--no-amp")
        cmds.append(("transfer_resnet18", cmd))

    # Transfer CNN (MobileNetV3 Small)
    if want("transfer_mobilenet_v3_small"):
        cmd = [sys.executable, str(ROOT / "scripts/training/train_transfer_cnn.py"),
               "--model", "mobilenet_v3_small"]
        add_common_args(cmd, data_root=args.data_root, log_dir=args.log_dir,
                        output_dir=args.output_dir, epochs=args.epochs,
                        batch_size=args.batch_size, num_workers=args.num_workers)
        if args.amp is True:
            cmd.append("--amp")
        elif args.amp is False:
            cmd.append("--no-amp")
        cmds.append(("transfer_mobilenet_v3_small", cmd))

    # EfficientNet-B0
    if want("efficientnet_b0"):
        cmd = [sys.executable, str(ROOT / "scripts/training/train_efficientnet_b0.py")]
        add_common_args(cmd, data_root=args.data_root, log_dir=args.log_dir,
                        output_dir=args.output_dir, epochs=args.epochs,
                        batch_size=args.batch_size, num_workers=args.num_workers)
        cmds.append(("efficientnet_b0", cmd))

    return cmds


def run_all(commands: Iterable[Tuple[str, List[str]]], dry_run: bool) -> int:
    for name, cmd in commands:
        pretty = " ".join(shlex.quote(part) for part in cmd)
        print(f"\n[RUN] {name}: {pretty}")
        if dry_run:
            continue
        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode != 0:
            print(f"[ERROR] {name} failed with exit code {result.returncode}")
            return result.returncode
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train all configured models sequentially with one command."
    )
    parser.add_argument("--data_root", type=Path, default=Path("data3a"))
    parser.add_argument("--log_dir", type=Path, default=Path("training_logs"))
    parser.add_argument("--output_dir", type=Path, default=Path("models"))
    parser.add_argument("--epochs", type=int,
                        help="Override epochs for deep nets (baseline/transfer/efficientnet)")
    parser.add_argument("--batch_size", type=int,
                        help="Override batch size for deep nets and GBT features")
    parser.add_argument("--num_workers", type=int,
                        help="Override dataloader workers where applicable")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None,
                        help="Force on/off mixed precision for transfer models")
    parser.add_argument("--models", type=parse_list,
                        help="Comma-separated subset to run (default: all)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--order", type=parse_list,
                        help="Custom comma-separated order; falls back to a sensible default")
    args = parser.parse_args()

    args.include = args.order or DEFAULT_ORDER
    commands = build_commands(args)
    if not commands:
        print("No models selected; nothing to do.")
        sys.exit(0)

    exit_code = run_all(commands, dry_run=args.dry_run)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
