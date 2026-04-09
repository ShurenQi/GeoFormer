import argparse
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, Any, List, Optional

warnings.filterwarnings("ignore", category=FutureWarning)

BASE_CONFIG: Dict[str, Any] = {
    "data": None,
    "output": None,
    "experiment": "scratch_run01",
    "model": None,
    "num_classes": 1000,
    "batch_size": 256,
    "epochs": 300,
    "warmup_epochs": 5,
    "workers": 8,
    "opt": "adamw",
    "weight_decay": 0.05,
    "lr": 1.5e-3,
    "min_lr": 1e-6,
    "warmup_lr": 1e-6,
    "sched": "cosine",
    "mixup": 0.8,
    "cutmix": 1.0,
    "smoothing": 0.1,
    "drop_path": 0.1,
    "aa": "rand-m9-mstd0.5-inc1",
    "amp": True,
    "model_ema": True,
    "model_ema_decay": 0.99996,
    "log_interval": 50,
    "recovery_interval": 0,
    "input_size": (3, 224, 224),
}


def build_argv_from_cfg(cfg: Dict[str, Any]) -> List[str]:
    argv: List[str] = []
    data = cfg.get("data")
    if data:
        argv.append(str(data))
    for k, v in cfg.items():
        if k == "data":
            continue
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            if v:
                argv.append(flag)
        elif isinstance(v, (list, tuple)):
            argv.append(flag)
            argv.extend([str(x) for x in v])
        elif v is None:
            continue
        else:
            argv.extend([flag, str(v)])
    return argv


def _has_flag(user_args: List[str], flag_name: str) -> bool:
    prefix = flag_name + "="
    return (flag_name in user_args) or any(a.startswith(prefix) for a in user_args)


def find_latest_checkpoint(output_dir: str, experiment: str) -> str:
    ckpt_dir = Path(output_dir) / experiment
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint dir not found: {ckpt_dir}")
    ckpt_main = ckpt_dir / "checkpoint.pth.tar"
    if ckpt_main.is_file():
        return str(ckpt_main)
    pat = re.compile(r"^checkpoint-(\d+)\.pth\.tar$")
    candidates = []
    for p in ckpt_dir.iterdir():
        if not p.is_file():
            continue
        m = pat.match(p.name)
        if m:
            candidates.append((int(m.group(1)), p))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in: {ckpt_dir}")
    candidates.sort(key=lambda x: x[0])
    return str(candidates[-1][1])


POOLFORMER_DROP_PATH = {
    "s12": 0.1,
    "s24": 0.1,
    "s36": 0.2,
    "m36": 0.3,
    "m48": 0.4,
}


def infer_scale_from_model_name(model_name: str) -> Optional[str]:
    m = re.search(r"geometaformer_(s12|s24|s36|m36|m48)_", model_name)
    return m.group(1) if m else None


def poolformer_lr_rule(batch_size: int) -> float:
    return float(batch_size) / 1024.0 * 1e-3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Train GeoMetaFormer (main + ablations), keep base recipe stable")
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--experiment", type=str, default=BASE_CONFIG["experiment"])
    p.add_argument(
        "--preset",
        type=str,
        default="auto",
        choices=["auto", "s12", "s24", "s36", "m36", "m48", "none"],
    )
    p.add_argument(
        "--lr-rule",
        type=str,
        default="none",
        choices=["none", "poolformer"],
    )
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--drop-path", type=float, default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--warmup-epochs", type=int, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--no-auto-resume", action="store_true")
    p.add_argument("--print-only", action="store_true")
    args, unknown = p.parse_known_args()
    args._unknown = unknown
    return args


def main():
    args = parse_args()

    import timm.models.geometaformer  # noqa: F401
    import timm.models.geometaformer_ablations  # noqa: F401

    cfg = dict(BASE_CONFIG)
    cfg["data"] = args.data
    cfg["model"] = args.model
    cfg["output"] = args.output
    cfg["experiment"] = args.experiment

    if args.preset == "none":
        scale = None
    elif args.preset == "auto":
        scale = infer_scale_from_model_name(args.model)
    else:
        scale = args.preset

    if scale is not None and scale in POOLFORMER_DROP_PATH:
        cfg["drop_path"] = POOLFORMER_DROP_PATH[scale]

    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.drop_path is not None:
        cfg["drop_path"] = args.drop_path

    if args.lr is not None:
        cfg["lr"] = args.lr
    elif args.lr_rule == "poolformer":
        cfg["lr"] = poolformer_lr_rule(int(cfg["batch_size"]))

    if args.workers is not None:
        cfg["workers"] = args.workers
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.warmup_epochs is not None:
        cfg["warmup_epochs"] = args.warmup_epochs
    if args.weight_decay is not None:
        cfg["weight_decay"] = args.weight_decay

    cfg_args = build_argv_from_cfg(cfg)
    user_args = list(args._unknown)

    if (not args.no_auto_resume) and (not _has_flag(user_args, "--resume")):
        ckpt_dir = Path(cfg["output"]) / cfg["experiment"]
        if ckpt_dir.exists():
            try:
                latest = find_latest_checkpoint(cfg["output"], cfg["experiment"])
                user_args += ["--resume", latest]
                print(f"[auto-resume] Using latest checkpoint: {latest}")
            except FileNotFoundError:
                print("[auto-resume] No checkpoint found, starting from scratch.")
        else:
            print("[auto-resume] Output dir not found, starting from scratch.")

    sys.argv = [sys.argv[0]] + cfg_args + user_args
    print(f"Running command:\n  python {' '.join(sys.argv)}\n")

    if args.print_only:
        return

    from train import main as timm_train
    timm_train()


if __name__ == "__main__":
    main()
