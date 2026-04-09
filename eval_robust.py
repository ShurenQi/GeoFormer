#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import timm
import timm.models.geometaformer
import timm.models.geometaformer_ablations
import timm.models.identity_rand_former
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder

CORRUPTIONS = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
]

ALEXNET_MEAN_ERRORS = {
    "gaussian_noise": 88.6,
    "shot_noise": 89.4,
    "impulse_noise": 92.2,
    "defocus_blur": 81.9,
    "glass_blur": 82.6,
    "motion_blur": 78.5,
    "zoom_blur": 79.8,
    "snow": 86.6,
    "frost": 82.6,
    "fog": 81.9,
    "brightness": 56.4,
    "contrast": 85.3,
    "elastic_transform": 64.6,
    "pixelate": 71.7,
    "jpeg_compression": 60.6,
}


def init_distributed():
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if distributed:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return distributed, rank, world_size, local_rank, device


@torch.inference_mode()
def validate_topk(model, loader, device, distributed=False, allowed_classes=None):
    model.eval()
    top1 = 0
    top5 = 0
    n = 0

    allowed = None
    if allowed_classes is not None:
        allowed = torch.as_tensor(sorted(list(allowed_classes)), device=device, dtype=torch.long)

    for images, target in loader:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        out = model(images)

        if allowed is not None:
            out = out.index_select(1, allowed)
            _, pred_pos = out.topk(5, 1, True, True)
            pred = allowed[pred_pos]
        else:
            _, pred = out.topk(5, 1, True, True)

        n += target.size(0)
        top1 += (pred[:, 0] == target).sum().item()
        top5 += (pred == target.view(-1, 1)).any(dim=1).sum().item()

    if distributed:
        t = torch.tensor([top1, top5, n], device=device, dtype=torch.long)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        top1, top5, n = (int(t[0].item()), int(t[1].item()), int(t[2].item()))

    return {"top1": 100.0 * top1 / n, "top5": 100.0 * top5 / n, "n": n}


def load_checkpoint_prefer_ema(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "state_dict_ema" in ckpt and ckpt["state_dict_ema"]:
        sd = ckpt["state_dict_ema"]
        source = "state_dict_ema"
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
        source = "state_dict"
    elif isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        sd = ckpt["model"]
        source = "model"
    elif isinstance(ckpt, dict):
        sd = ckpt
        source = "raw_dict"
    else:
        raise KeyError(f"Checkpoint format not recognized: {ckpt_path}")
    new_sd = {}
    for k, v in sd.items():
        k2 = k
        if k2.startswith("module."):
            k2 = k2[len("module.") :]
        if k2.startswith("model."):
            k2 = k2[len("model.") :]
        new_sd[k2] = v
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    return source, missing, unexpected


def load_wnid_to_idx(class_map_json=""):
    if class_map_json:
        obj = json.loads(Path(class_map_json).read_text())
        if isinstance(obj, dict):
            keys = list(obj.keys())
            if keys and str(keys[0]).isdigit():
                wnid_to_idx = {}
                for k, v in obj.items():
                    idx = int(k)
                    if isinstance(v, (list, tuple)) and len(v) >= 1:
                        wnid = v[0]
                    elif isinstance(v, dict) and "wnid" in v:
                        wnid = v["wnid"]
                    else:
                        raise ValueError("class_map_json format not recognized")
                    wnid_to_idx[str(wnid)] = idx
                return wnid_to_idx
            return {str(k): int(v) for k, v in obj.items()}
        raise ValueError("class_map_json must be a dict")
    try:
        from timm.data.imagenet_info import ImageNetInfo

        info = ImageNetInfo()
        wnids = None
        for attr in ("wnids", "wnid"):
            if hasattr(info, attr):
                wnids = getattr(info, attr)
                break
        if wnids is None:
            for attr in ("index_to_wnid", "idx_to_wnid"):
                if hasattr(info, attr):
                    m = getattr(info, attr)
                    if callable(m):
                        m = m()
                    if isinstance(m, dict):
                        wnids = [m[i] for i in range(1000)]
                        break
        if wnids is None:
            if hasattr(info, "synsets"):
                s = getattr(info, "synsets")
                if isinstance(s, (list, tuple)) and len(s) >= 1000:
                    wnids = list(s)[:1000]
        if wnids is None or len(wnids) < 1000:
            raise RuntimeError("Could not infer wnids from timm ImageNetInfo")
        return {str(wnid): i for i, wnid in enumerate(wnids[:1000])}
    except Exception as e:
        raise RuntimeError(
            "Failed to get ImageNet wnid->idx mapping. Please provide --class_map_json. "
            f"Original error: {e}"
        )


def remap_imagefolder_targets(dataset, wnid_to_idx):
    classes = dataset.classes
    class_to_target = {}
    for i, cls_name in enumerate(classes):
        if cls_name not in wnid_to_idx:
            raise KeyError(f"Class folder '{cls_name}' not found in wnid_to_idx mapping")
        class_to_target[i] = int(wnid_to_idx[cls_name])
    samples = []
    for path, old_y in dataset.samples:
        samples.append((path, class_to_target[old_y]))
    dataset.samples = samples
    dataset.imgs = samples
    dataset.targets = [y for _, y in samples]
    dataset.class_to_idx = {k: int(wnid_to_idx[k]) for k in classes}
    dataset.classes = classes
    return dataset


def make_loader(data_root, model, batch_size, workers, distributed, rank, world_size, wnid_to_idx=None):
    cfg = resolve_data_config({}, model=model)
    transform = create_transform(**cfg, is_training=False)
    dataset = ImageFolder(str(data_root), transform=transform)
    if wnid_to_idx is not None:
        dataset = remap_imagefolder_targets(dataset, wnid_to_idx)
    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True if workers > 0 else False,
    )
    return loader, cfg, len(dataset)


def compute_mce(c_res, alexnet_mean_errors):
    ces = {}
    for corr in CORRUPTIONS:
        errs = []
        for sev in ["1", "2", "3", "4", "5"]:
            acc = c_res[corr][sev]["top1"]
            errs.append(100.0 - acc)
        mean_err = sum(errs) / len(errs)
        denom = float(alexnet_mean_errors[corr])
        ces[corr] = 100.0 * (mean_err / denom)
    mce = sum(ces.values()) / len(ces)
    return mce, ces


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--imagenet_a", type=str, default="")
    p.add_argument("--imagenet_r", type=str, default="")
    p.add_argument("--imagenet_c", type=str, default="")
    p.add_argument("--imagenet_sk", type=str, default="")
    p.add_argument("--imagenet_sketch", type=str, default="")
    p.add_argument("--class_map_json", type=str, default="")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--output_json", type=str, default="eval_arc_results.json")
    p.add_argument("--alexnet_mce_json", type=str, default="")
    args = p.parse_args()

    if args.imagenet_sk and args.imagenet_sketch:
        raise ValueError("Please provide only one of --imagenet_sk or --imagenet_sketch")

    imagenet_sk = args.imagenet_sk if args.imagenet_sk else args.imagenet_sketch

    distributed, rank, world_size, local_rank, device = init_distributed()

    model = timm.create_model(args.model, pretrained=args.pretrained, num_classes=1000)
    model.to(device)

    source = "timm_pretrained" if args.pretrained else ""
    missing = []
    unexpected = []
    if args.checkpoint:
        source, missing, unexpected = load_checkpoint_prefer_ema(model, args.checkpoint)

    if rank == 0:
        if args.pretrained:
            print("[ckpt] timm pretrained: true", flush=True)
        if args.checkpoint:
            print(f"[ckpt] loaded from: {source}", flush=True)
            if missing:
                print(f"[ckpt] missing keys: {len(missing)}", flush=True)
            if unexpected:
                print(f"[ckpt] unexpected keys: {len(unexpected)}", flush=True)

    wnid_to_idx = None
    if args.imagenet_a or args.imagenet_r or args.imagenet_c or imagenet_sk:
        wnid_to_idx = load_wnid_to_idx(args.class_map_json)

    alexnet_mean_errors = dict(ALEXNET_MEAN_ERRORS)
    if args.alexnet_mce_json:
        obj = json.loads(Path(args.alexnet_mce_json).read_text())
        for k, v in obj.items():
            alexnet_mean_errors[str(k)] = float(v)

    results = None
    if rank == 0:
        results = {
            "model": args.model,
            "pretrained": bool(args.pretrained),
            "checkpoint": args.checkpoint,
            "weights_source": source,
            "class_map_json": args.class_map_json,
            "world_size": world_size,
        }

    if args.imagenet_a:
        loader, cfg, n = make_loader(
            args.imagenet_a, model, args.batch_size, args.workers, distributed, rank, world_size, wnid_to_idx
        )
        if distributed and loader.sampler is not None:
            loader.sampler.set_epoch(0)
        allowed = set(loader.dataset.targets)
        res = validate_topk(model, loader, device=device, distributed=distributed, allowed_classes=allowed)
        if rank == 0:
            results["imagenet_a"] = {"path": args.imagenet_a, "n": n, **res, "restricted": True, "data_cfg": cfg}
            print(f"[IN-A] top1={res['top1']:.3f} top5={res['top5']:.3f} n={res['n']}", flush=True)

    if args.imagenet_r:
        loader, cfg, n = make_loader(
            args.imagenet_r, model, args.batch_size, args.workers, distributed, rank, world_size, wnid_to_idx
        )
        if distributed and loader.sampler is not None:
            loader.sampler.set_epoch(0)
        allowed = set(loader.dataset.targets)
        res = validate_topk(model, loader, device=device, distributed=distributed, allowed_classes=allowed)
        if rank == 0:
            results["imagenet_r"] = {"path": args.imagenet_r, "n": n, **res, "restricted": True, "data_cfg": cfg}
            print(f"[IN-R] top1={res['top1']:.3f} top5={res['top5']:.3f} n={res['n']}", flush=True)

    if imagenet_sk:
        loader, cfg, n = make_loader(
            imagenet_sk, model, args.batch_size, args.workers, distributed, rank, world_size, wnid_to_idx
        )
        if distributed and loader.sampler is not None:
            loader.sampler.set_epoch(0)
        allowed = set(loader.dataset.targets)
        res = validate_topk(model, loader, device=device, distributed=distributed, allowed_classes=allowed)
        if rank == 0:
            results["imagenet_sk"] = {"path": imagenet_sk, "n": n, **res, "restricted": True, "data_cfg": cfg}
            print(f"[IN-SK] top1={res['top1']:.3f} top5={res['top5']:.3f} n={res['n']}", flush=True)

    if args.imagenet_c:
        c_root = Path(args.imagenet_c)
        c_res = {}
        for corr in CORRUPTIONS:
            corr_res = {}
            for sev in [1, 2, 3, 4, 5]:
                sev_root = c_root / corr / str(sev)
                if not sev_root.exists():
                    raise FileNotFoundError(f"Missing: {sev_root}")
                loader, cfg, n = make_loader(
                    sev_root, model, args.batch_size, args.workers, distributed, rank, world_size, wnid_to_idx
                )
                if distributed and loader.sampler is not None:
                    loader.sampler.set_epoch(0)
                res = validate_topk(model, loader, device=device, distributed=distributed, allowed_classes=None)
                if rank == 0:
                    corr_res[str(sev)] = {"n": n, **res}
                    print(f"[IN-C] {corr} s={sev} top1={res['top1']:.3f}", flush=True)
            if rank == 0:
                c_res[corr] = corr_res

        if rank == 0:
            mce, ce_per_corr = compute_mce(c_res, alexnet_mean_errors)
            results["imagenet_c"] = {
                "path": args.imagenet_c,
                "per_corruption": c_res,
                "alexnet_mean_errors": alexnet_mean_errors,
                "ce_per_corruption": ce_per_corr,
                "mCE": mce,
            }
            print(f"[IN-C] mCE (AlexNet normalized) = {mce:.3f}", flush=True)

    if distributed:
        dist.barrier()

    if rank == 0:
        Path(args.output_json).write_text(json.dumps(results, indent=2))
        print(f"[done] wrote: {args.output_json}", flush=True)

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
