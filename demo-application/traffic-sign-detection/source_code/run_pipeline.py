import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_step(command: list[str], dry_run: bool) -> int:
    cmd_str = " ".join(command)
    print(f"\n[STEP] {cmd_str}")
    if dry_run:
        return 0

    result = subprocess.run(command, cwd=ROOT)
    if result.returncode != 0:
        print(f"[ERROR] Step failed with code {result.returncode}")
    return result.returncode


def build_commands(args: argparse.Namespace) -> list[list[str]]:
    py = sys.executable

    commands = []

    if args.phase in {"all", "phase1"}:
        commands.append(
            [
                py,
                "source_code/data_audit.py",
                "--data",
                args.data,
                "--split",
                "train",
                "--output",
                "data_audit_report_train.json",
                "--plot-dir",
                "audit_plots/train",
            ]
        )
        commands.append(
            [
                py,
                "source_code/data_balance.py",
                "--data",
                args.data,
                "--split",
                "all",
                "--ensure-splits",
                "--output-root",
                args.output_root,
                "--min-instances",
                str(args.min_instances),
            ]
        )

    if args.phase in {"all", "phase2"}:
        if args.sources_json:
            commands.append(
                [
                    py,
                    "source_code/download_external_data.py",
                    "--sources-json",
                    args.sources_json,
                    "--dataset-out",
                    args.output_root,
                ]
            )
        if args.run_pseudo_label:
            cmd = [
                py,
                "source_code/smart_augment.py",
                "--input-dir",
                args.external_image_dir,
                "--out-images",
                f"{args.output_root}/images/train",
                "--out-labels",
                f"{args.output_root}/labels/train",
                "--conf",
                str(args.pseudo_conf),
            ]
            if args.class_map_json:
                cmd.extend(["--class-map-json", args.class_map_json])
            commands.append(cmd)

    if args.phase in {"all", "phase3"}:
        commands.append(
            [
                py,
                "source_code/train_yolo26.py",
                "--model",
                args.model,
                "--data",
                args.data_v2,
                "--name",
                args.run_name,
                "--epochs",
                str(args.epochs),
                "--imgsz",
                str(args.imgsz),
                "--batch",
                str(args.batch),
                "--device",
                args.device,
            ]
        )

    if args.phase in {"all", "phase4"}:
        if args.run_tune:
            commands.append(
                [
                    py,
                    "source_code/tune_yolo26.py",
                    "--mode",
                    "tune",
                    "--weights",
                    args.tune_weights,
                    "--data",
                    args.data_v2,
                    "--imgsz",
                    str(args.imgsz),
                    "--device",
                    args.device,
                    "--tune-epochs",
                    str(args.tune_epochs),
                    "--iterations",
                    str(args.iterations),
                ]
            )
        if args.run_final:
            commands.append(
                [
                    py,
                    "source_code/tune_yolo26.py",
                    "--mode",
                    "final",
                    "--model",
                    args.model,
                    "--data",
                    args.data_v2,
                    "--cfg",
                    args.tune_cfg,
                    "--name",
                    args.final_name,
                    "--imgsz",
                    str(args.imgsz),
                    "--device",
                    args.device,
                    "--final-epochs",
                    str(args.final_epochs),
                ]
            )

    if args.phase in {"all", "phase5"}:
        commands.append(
            [
                py,
                "source_code/evaluate.py",
                "--baseline",
                args.baseline_weights,
                "--candidate",
                args.eval_candidate_weights,
                "--data",
                args.data_v2,
                "--imgsz",
                str(args.imgsz),
                "--device",
                args.device,
                "--warmup",
                str(args.warmup),
            ]
        )

    return commands


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO26 traffic-sign pipeline by phase")
    parser.add_argument("--phase", default="all", choices=["all", "phase1", "phase2", "phase3", "phase4", "phase5"])
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue even if one step fails")

    parser.add_argument("--data", default="data.yaml")
    parser.add_argument("--data-v2", default="data_v2.yaml")
    parser.add_argument("--output-root", default="datasets_v2")
    parser.add_argument("--min-instances", type=int, default=300)

    parser.add_argument("--sources-json", default="")
    parser.add_argument("--external-image-dir", default="external_data/images")
    parser.add_argument("--run-pseudo-label", action="store_true")
    parser.add_argument("--pseudo-conf", type=float, default=0.7)
    parser.add_argument("--class-map-json", default="")

    parser.add_argument("--model", default="yolo26s.pt")
    parser.add_argument("--run-name", default="yolo26s_traffic_v1")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")

    parser.add_argument("--run-tune", action="store_true")
    parser.add_argument("--run-final", action="store_true")
    parser.add_argument("--tune-weights", default="runs/detect/yolo26s_traffic_v1/weights/best.pt")
    parser.add_argument("--tune-cfg", default="runs/detect/tune/best_hyperparameters.yaml")
    parser.add_argument("--tune-epochs", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--final-name", default="yolo26s_traffic_final")
    parser.add_argument("--final-epochs", type=int, default=200)

    parser.add_argument("--baseline-weights", default="runs/detect/traffic_detect_run3/weights/best.pt")
    parser.add_argument("--eval-candidate-weights", default="runs/detect/yolo26s_traffic_final/weights/best.pt")
    parser.add_argument("--warmup", type=int, default=10)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    commands = build_commands(args)

    if not commands:
        print("[WARN] No commands generated for selected options")
        return

    for command in commands:
        code = run_step(command, dry_run=args.dry_run)
        if code != 0 and not args.continue_on_error:
            raise SystemExit(code)

    print("\n[INFO] Pipeline execution completed")


if __name__ == "__main__":
    main()
