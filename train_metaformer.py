import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

CONFIG = {
    "data_dir": "/home/shurenqi/datasets/imagenet1k/",
    "num_classes": 1000,
    "batch_size": 256,
    "epochs": 300,
    "warmup_epochs": 5,
    "workers": 8,
    "opt": "adamw",
    "weight_decay": 0.05,
    "min_lr": 1e-6,
    "warmup_lr": 1e-6,
    "sched": "cosine",
    "mixup": 0.8,
    "cutmix": 1.0,
    "smoothing": 0.1,
    "aa": "rand-m9-mstd0.5-inc1",
    "amp": True,
    "log_interval": 50,
    "recovery_interval": 0,
    "input_size": (3, 224, 224),
}

def _has_flag(user_args, flag_name: str) -> bool:
    prefix = flag_name + "="
    return (flag_name in user_args) or any(a.startswith(prefix) for a in user_args)

def _get_flag_value(user_args, flag_name: str, default=None):
    prefix = flag_name + "="
    for i, a in enumerate(user_args):
        if a == flag_name and i + 1 < len(user_args):
            return user_args[i + 1]
        if a.startswith(prefix):
            return a[len(prefix):]
    return default

def build_default_argv(cfg: dict, user_args):
    argv = []
    for k, v in cfg.items():
        flag = "--" + k.replace("_", "-")
        no_flag = "--no-" + k.replace("_", "-")


        if _has_flag(user_args, flag) or _has_flag(user_args, no_flag):
            continue

        if isinstance(v, bool):
            if v:
                argv.append(flag)
        elif isinstance(v, (list, tuple)):
            argv.append(flag)
            argv.extend(map(str, v))
        else:
            argv.extend([flag, str(v)])
    return argv

def _normalize_data_flag(user_args):
    out = []
    for a in user_args:
        if a == "--data":
            out.append("--data-dir")
        elif a.startswith("--data="):
            out.append("--data-dir=" + a.split("=", 1)[1])
        else:
            out.append(a)
    return out

def _model_needs_custom_registration(model_name: str) -> bool:
    if not model_name:
        return False
    model_name = model_name.lower()
    return model_name.startswith("randformer_") or model_name.startswith("identityformer_")

def _ensure_custom_models_imported_if_needed(user_args, cfg):
    model_name = _get_flag_value(user_args, "--model", default=cfg.get("model", ""))
    if not _model_needs_custom_registration(model_name):
        return
    try:
        import timm.models.identity_rand_former  # noqa: F401
        print(f"[model-registry] Imported timm.models.identity_rand_former for model={model_name}")
    except Exception as e:
        raise RuntimeError(
            "Failed to import timm.models.identity_rand_former. "
            "Confirm identity_rand_former.py is under timm/models/ and you are using this timm checkout."
        ) from e

    if model_name.lower().startswith("randformer_"):
        in0 = _get_flag_value(user_args, "--input-size", default=None)
        in_size = cfg.get("input_size", None) if in0 is None else None
        if isinstance(in_size, (list, tuple)) and tuple(in_size) != (3, 224, 224):
            raise ValueError(f"RandFormer fixed-224 requires input_size=(3,224,224), but got {in_size}.")

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

def main():
    user_args = _normalize_data_flag(sys.argv[1:])


    _ensure_custom_models_imported_if_needed(user_args, CONFIG)


    cfg_args = build_default_argv(CONFIG, user_args)

    merged_args = cfg_args + user_args

    out_dir = _get_flag_value(merged_args, "--output", default=None)
    exp = _get_flag_value(merged_args, "--experiment", default=None)

    if out_dir and exp and not _has_flag(merged_args, "--resume"):
        ckpt_dir = Path(out_dir) / exp
        if ckpt_dir.exists():
            try:
                latest = find_latest_checkpoint(out_dir, exp)
                merged_args = merged_args + ["--resume", latest]
                print(f"[auto-resume] Using latest checkpoint: {latest}")
            except FileNotFoundError:
                pass

    sys.argv = [sys.argv[0]] + merged_args
    print(f"Running command: {' '.join(sys.argv)}")

    from train import main as timm_train
    timm_train()

if __name__ == "__main__":
    main()

