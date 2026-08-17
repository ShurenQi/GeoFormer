# GeoFormer

## 1. Paper information

**Title:** Geometric Invariance Priors in Vision Transformers: Principles and Costs  
**Authors:** Shuren Qi, Yushu Zhang, Yuming Fang, Xiaochun Cao, and Fenglei Fan  
**Keywords:** Vision Transformers, geometric invariance, inductive priors

### Abstract

In vision Transformers, geometric invariance priors expose a long-standing tension: they are expected to improve efficiency and robustness, yet empirically they conflict with discriminative and scalable learning. A clear theoretical characterization of this tension is still lacking. To our knowledge, this paper gives the first systematic study to characterize the principled form and structural cost of geometric invariance priors in vision Transformers, offering the geometric interpretation and historical continuity of MetaFormer-like designs.

1. **Principles.** We use *localized Euclidean-similarity transformations* as a visually natural model of geometric compatibility: the same local visual structure may appear at different locations, orientations, and scales, while a generic learning rule should not depend on these local coordinate choices. We prove that, in a generic and scalable learning setting where feature channels carry *no* prescribed geometric structure, requiring localized-similarity compatibility forces any regular local operator to collapse to *shared pointwise channel mixing*. In the linear case, this result reinterprets the ubiquitous $1\times1$ convolution in vision networks: beyond its engineering role in channel projection and computational efficiency, it can also be viewed as the principled form of geometry-compatible generic learning.

2. **Costs.** This pointwise collapse reveals the structural cost of invariance priors: geometry-compatible generic learning *cannot* simultaneously provide spatial discrimination. Thus, if a backbone is to preserve geometric compatibility while continuously renewing discriminative spatial information throughout the hierarchy, spatial interaction must be assigned to geometry-aware spatial mixing. The main generic learnable capacity and semantic recombination can then remain in pointwise channel mixing. From this perspective, the *MetaFormer-like separation* between spatial and channel mixing is a *structural trade-off* among invariance, discriminability, and scalable learning.

Guided by the principles and costs, we propose GeoFormer as a *unified architecture* for introducing geometric invariance priors into vision Transformers. It extends MetaFormer-like designs to a *richer spectrum* of invariance priors, from heuristic priors, to highly designed localized-similarity priors, and to fully data-adaptive convolution priors. GeoFormer allows us to achieve concrete trade-offs among invariance, discriminability, and scalability more flexibly in practical tasks, supported by substantial numerical evidence on CIFAR-100 and ImageNet.

## 2. Core files and registered models

### Model definitions

| File | Description |
| --- | --- |
| `timm/models/geoformer.py` | Geometric basis responses, GeoMixer, and GeoFormer |
| `timm/models/gaformer.py` | Inverted-bottleneck GeoBlock and GAFormer |
| `timm/models/paformer.py` | Pooling counterpart and PAFormer |
| `timm/models/identity_rand_former.py` | IdentityFormer and RandFormer priors |
| `timm/models/metaformer.py` | MetaFormer, PoolFormer, PoolFormerV2, ConvFormer, and CAFormer families |
| `timm/models/geoformer_ablations.py` | Multi-scale and higher-order GeoMixer variants |
| `timm/models/_geoformer_common.py` | Model scales and shared construction |
| `timm/models/__init__.py` | Model registration |

### Registered models

| Family | Models |
| --- | --- |
| GeoFormer | `geoformer_s12`, `geoformer_s24`, `geoformer_s36`, `geoformer_m36`, `geoformer_m48` |
| GeoFormer ablations | `geoformer_s12_multiscale`, `geoformer_m48_multiscale`, `geoformer_s12_higher_order`, `geoformer_m48_higher_order` |
| GAFormer | `gaformer_s18`, `gaformer_s36`, `gaformer_m36` |
| PAFormer | `paformer_s18`, `paformer_s36`, `paformer_m36` |
| IdentityFormer | `identityformer_s12`, `identityformer_s24`, `identityformer_s36`, `identityformer_m36`, `identityformer_m48` |
| RandFormer | `randformer_s12`, `randformer_s24`, `randformer_s36`, `randformer_m36`, `randformer_m48` |
| PoolFormer | `poolformer_s12`, `poolformer_s24`, `poolformer_s36`, `poolformer_m36`, `poolformer_m48` |
| PoolFormerV2 | `poolformerv2_s12`, `poolformerv2_s24`, `poolformerv2_s36`, `poolformerv2_m36`, `poolformerv2_m48` |
| ConvFormer | `convformer_s18`, `convformer_s36`, `convformer_m36` |
| CAFormer | `caformer_s18`, `caformer_s36`, `caformer_m36` |

### Execution files

| File | Description |
| --- | --- |
| `toy_example.py` | GAFormer/PAFormer construction, training, evaluation, and comparison |
| `timm/train_imagenet.py` | ImageNet training entry for all registered models |
| `timm/train.py` | ImageNet training engine |
| `timm/evaluate_robustness.py` | ImageNet-A robustness evaluation |
| `timm/data/_info/` | ImageNet class metadata |
| `timm/data/`, `timm/layers/`, `timm/loss/`, `timm/optim/`, `timm/scheduler/`, `timm/utils/` | Bundled timm runtime |

## 3. Quick start, case, and toy example

Run all commands from the repository root.

```python
import torch
import timm

model = timm.create_model("geoformer_s12", pretrained=False, num_classes=1000)
model.eval()

with torch.no_grad():
    logits = model(torch.randn(1, 3, 224, 224))

print(logits.shape)
```

Run the self-contained GAFormer/PAFormer example on synthetically generated data, including model construction, toy training, evaluation, and comparison:

```bash
python toy_example.py
```

## 4. Official commands

### Training and direct validation

The single training entry implements the paper's 300-epoch ImageNet-1K recipe for every registered model. With eight processes and the default per-process batch size of 256, the global batch size is 2048. The validation split is evaluated directly after each training epoch.

```bash
torchrun --standalone --nproc_per_node=8 -m timm.train_imagenet \
  --data /path/to/imagenet \
  --model geoformer_s12 \
  --output ./output \
  --experiment geoformer_s12
```

Change `--model` and `--experiment` together to train any model in the registered list, for example `gaformer_s18`, `paformer_s18`, `poolformer_s12`, `identityformer_s12`, `randformer_s12`, `convformer_s18`, or `caformer_s18`.

### Transferability and robustness evaluation

Evaluate a trained checkpoint on ImageNet-A:

```bash
python -m timm.evaluate_robustness \
  --model geoformer_s12 \
  --checkpoint ./output/geoformer_s12/checkpoint.pth.tar \
  --imagenet-a /path/to/imagenet-a \
  --output-json geoformer_s12_imagenet_a.json
```

## 5. Environment and statements

### Environment

- Python 3.10+
- PyTorch, torchvision, and PyYAML
- No external timm installation

```bash
pip install torch torchvision pyyaml
```

### Statements

We gratefully acknowledge Ross Wightman and all contributors to the [`timm`](https://github.com/huggingface/pytorch-image-models) library. The bundled runtime is derived from timm 0.9.16; its original notices and Apache License 2.0 terms are retained in `timm/LICENSE`.

Copyright (c) 2026 Shuren Qi, Yushu Zhang, Yuming Fang, Xiaochun Cao, and Fenglei Fan. All rights reserved. Third-party components remain subject to their respective notices and licenses.
