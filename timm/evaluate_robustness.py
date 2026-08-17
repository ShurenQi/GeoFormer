from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.distributed as dist
from torch import Tensor
from torch.utils.data import DataLoader, Sampler
from torchvision.datasets import ImageFolder

import timm
from timm.data import ImageNetInfo, resolve_data_config
from timm.data.transforms_factory import create_transform


class ExactDistributedSampler(Sampler[int]):
    def __init__(self, dataset_size: int, rank: int, world_size: int):
        self.dataset_size = int(dataset_size)
        self.rank = int(rank)
        self.world_size = int(world_size)

    def __iter__(self):
        return iter(range(self.rank, self.dataset_size, self.world_size))

    def __len__(self) -> int:
        remaining = self.dataset_size - self.rank
        return max(0, (remaining + self.world_size - 1) // self.world_size)


def initialize_distributed() -> tuple[bool, int, int, torch.device]:
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if not distributed:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return False, 0, 1, device

    has_cuda = torch.cuda.is_available()
    dist.init_process_group(backend="nccl" if has_cuda else "gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if has_cuda:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return True, rank, world_size, device


@torch.inference_mode()
def evaluate_topk(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    distributed: bool,
    allowed_classes: set[int],
) -> dict[str, float | int]:
    model.eval()
    allowed = torch.tensor(
        sorted(allowed_classes),
        device=device,
        dtype=torch.long,
    )
    top1_count = 0
    top5_count = 0
    sample_count = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        restricted_logits = model(images).index_select(1, allowed)
        top_k = min(5, restricted_logits.shape[1])
        predicted_positions = restricted_logits.topk(top_k, dim=1).indices
        predictions = allowed[predicted_positions]

        sample_count += targets.shape[0]
        top1_count += int((predictions[:, 0] == targets).sum())
        top5_count += int((predictions == targets.view(-1, 1)).any(dim=1).sum())

    if distributed:
        totals = torch.tensor(
            [top1_count, top5_count, sample_count],
            device=device,
            dtype=torch.long,
        )
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        top1_count, top5_count, sample_count = (
            int(totals[0]),
            int(totals[1]),
            int(totals[2]),
        )

    return {
        "top1": 100.0 * top1_count / sample_count,
        "top5": 100.0 * top5_count / sample_count,
        "n": sample_count,
    }


def load_checkpoint_prefer_ema(
    model: torch.nn.Module,
    checkpoint_path: str,
) -> str:
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Checkpoint format not recognized: {checkpoint_path}")

    if isinstance(checkpoint.get("state_dict_ema"), Mapping):
        state_dict = checkpoint["state_dict_ema"]
        source = "state_dict_ema"
    elif isinstance(checkpoint.get("model_ema"), Mapping):
        state_dict = checkpoint["model_ema"]
        source = "model_ema"
    elif isinstance(checkpoint.get("state_dict"), Mapping):
        state_dict = checkpoint["state_dict"]
        source = "state_dict"
    elif isinstance(checkpoint.get("model"), Mapping):
        state_dict = checkpoint["model"]
        source = "model"
    else:
        state_dict = checkpoint
        source = "raw_dict"

    cleaned_state = {}
    for key, value in state_dict.items():
        clean_key = key
        if clean_key.startswith("module."):
            clean_key = clean_key[len("module.") :]
        if clean_key.startswith("model."):
            clean_key = clean_key[len("model.") :]
        cleaned_state[clean_key] = value

    model.load_state_dict(cleaned_state, strict=True)
    return source


def load_wnid_to_index(class_map_json: str = "") -> dict[str, int]:
    if not class_map_json:
        labels = ImageNetInfo().label_names()
        if len(labels) != 1000:
            raise RuntimeError(
                f"Expected 1,000 ImageNet labels, received {len(labels)}"
            )
        return {wnid: index for index, wnid in enumerate(labels)}

    class_map = json.loads(Path(class_map_json).read_text(encoding="utf-8"))
    if not isinstance(class_map, dict):
        raise ValueError("class-map JSON must contain an object")

    keys = list(class_map)
    if not keys or not str(keys[0]).isdigit():
        return {str(wnid): int(index) for wnid, index in class_map.items()}

    wnid_to_index = {}
    for raw_index, value in class_map.items():
        index = int(raw_index)
        if isinstance(value, (list, tuple)) and value:
            wnid = value[0]
        elif isinstance(value, dict) and "wnid" in value:
            wnid = value["wnid"]
        else:
            raise ValueError("class-map JSON format not recognized")
        wnid_to_index[str(wnid)] = index
    return wnid_to_index


def remap_targets(
    dataset: ImageFolder,
    wnid_to_index: dict[str, int],
) -> ImageFolder:
    folder_index_to_target = {}
    for folder_index, wnid in enumerate(dataset.classes):
        if wnid not in wnid_to_index:
            raise KeyError(f"Class folder '{wnid}' is not an ImageNet-1K WNID")
        folder_index_to_target[folder_index] = wnid_to_index[wnid]

    samples = [
        (path, folder_index_to_target[folder_index])
        for path, folder_index in dataset.samples
    ]
    dataset.samples = samples
    dataset.imgs = samples
    dataset.targets = [target for _, target in samples]
    dataset.class_to_idx = {wnid: wnid_to_index[wnid] for wnid in dataset.classes}
    return dataset


def create_imagenet_a_loader(
    data_root: str,
    model: torch.nn.Module,
    batch_size: int,
    workers: int,
    *,
    distributed: bool,
    rank: int,
    world_size: int,
    wnid_to_index: dict[str, int],
) -> tuple[DataLoader, dict, int]:
    data_config = resolve_data_config({}, model=model)
    transform = create_transform(**data_config, is_training=False)
    dataset = ImageFolder(data_root, transform=transform)
    dataset = remap_targets(dataset, wnid_to_index)

    sampler = None
    if distributed:
        sampler = ExactDistributedSampler(
            len(dataset),
            rank,
            world_size,
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=workers > 0,
    )
    return loader, data_config, len(dataset)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a MetaFormer-family model on ImageNet-A"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument(
        "--imagenet-a",
        "--imagenet_a",
        dest="imagenet_a",
        required=True,
    )
    parser.add_argument(
        "--class-map-json",
        "--class_map_json",
        dest="class_map_json",
        default="",
    )
    parser.add_argument(
        "--batch-size",
        "--batch_size",
        dest="batch_size",
        type=int,
        default=256,
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--output-json",
        "--output_json",
        dest="output_json",
        default="imagenet_a_results.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    distributed, rank, world_size, device = initialize_distributed()

    model = timm.create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=1000,
    ).to(device)

    weight_source = "timm_pretrained" if args.pretrained else ""
    if args.checkpoint:
        weight_source = load_checkpoint_prefer_ema(model, args.checkpoint)

    if rank == 0:
        if args.pretrained:
            print("[checkpoint] using timm pretrained weights", flush=True)
        if args.checkpoint:
            print(f"[checkpoint] loaded {weight_source}", flush=True)

    wnid_to_index = load_wnid_to_index(args.class_map_json)
    loader, data_config, dataset_size = create_imagenet_a_loader(
        args.imagenet_a,
        model,
        args.batch_size,
        args.workers,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        wnid_to_index=wnid_to_index,
    )
    metrics = evaluate_topk(
        model,
        loader,
        device,
        distributed=distributed,
        allowed_classes=set(loader.dataset.targets),
    )

    if rank == 0:
        results = {
            "model": args.model,
            "pretrained": bool(args.pretrained),
            "checkpoint": args.checkpoint,
            "weights_source": weight_source,
            "class_map_json": args.class_map_json,
            "world_size": world_size,
            "imagenet_a": {
                "path": args.imagenet_a,
                "n": dataset_size,
                **metrics,
                "restricted": True,
                "data_config": data_config,
            },
        }
        Path(args.output_json).write_text(
            json.dumps(results, indent=2),
            encoding="utf-8",
        )
        print(
            f"[ImageNet-A] top1={metrics['top1']:.3f} "
            f"top5={metrics['top5']:.3f} n={metrics['n']}",
            flush=True,
        )
        print(f"[done] wrote {args.output_json}", flush=True)

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
