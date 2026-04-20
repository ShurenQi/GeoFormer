# GeoFormer

Official PyTorch implementation of **Structured Invariance for Vision Backbones: A Geometric Route to MetaFormer**.

## Overview

Invariance has long been a foundational prior in vision, but its role in modern vision backbones under scaling laws has become increasingly ambiguous. In practice, it is often learned implicitly through data, augmentation, and scale, while making it explicit is commonly viewed as restricting representational flexibility.

This paper studies the role of invariance in vision backbones through a geometric route to MetaFormer, centered on two questions: where invariance should reside in vision backbones, and whether explicit invariance can yield gains under scaling.

To answer these questions, we extend the theory of moments and moment invariants to modern backbones by formulating a unified learnable framework for global, local, and hierarchical invariance, in which the role of learnable operators becomes analytically explicit.

Within this framework, we establish a classification theorem showing that, under suitable assumptions, every admissible continuous local linear operator with scalar-like channels reduces to shared pointwise 1x1 convolution. This result reveals a principled role for 1x1 convolution in geometry-compatible learning.

Because the admissible operator is spatially blind, geometric structure must be maintained through the interaction of two roles throughout the hierarchy: geometry-aware spatial operators and generic pointwise channel operators. This compositional pattern provides a geometric interpretation of the MetaFormer decomposition into spatial and channel mixing.

Guided by the structured view, we instantiate **GeoFormer**, where geometric priors reside primarily in spatial mixing while channel mixing remains generic and pointwise. Experiments show that this structured design improves the invariance-discriminability trade-off and exhibits favorable scaling behavior.

## Files, Dependency, and Directory Structure

This repository is built on top of the `timm` / `pytorch-image-models` codebase and is **expected to follow the source-tree structure of `timm`**.

In other words, this project is **not intended to be used as a completely standalone flat folder**. To run the training and evaluation scripts correctly, the repository should be organized in a way that is compatible with the original `timm` directory layout, especially for model definition files under `timm/models/`.

A recommended directory structure is:

~~~text
pytorch-image-models/
├── train_geometaformer.py
├── train_metaformer.py
├── eval_robust.py
├── wnid_to_idx_1k.json
├── bash.txt
└── timm/
    └── models/
        ├── geometaformer.py
        ├── geometaformer_ablations.py
        ├── identity_rand_former.py
        └── metaformer.py
~~~

### Root directory

- `train_geometaformer.py`: training entry for GeoFormer models
- `train_metaformer.py`: training entry for MetaFormer-style baselines
- `eval_robust.py`: robustness evaluation on ImageNet-A / R / C / Sketch
- `wnid_to_idx_1k.json`: ImageNet-1K class mapping
- `bash.txt`: command examples used in experiments

### `timm/models/`

- `geometaformer.py`: GeoFormer model definitions
- `geometaformer_ablations.py`: ablation model definitions
- `identity_rand_former.py`: MetaFormer baselines, including IdentityFormer / RandFormer
- `metaformer.py`: MetaFormer baselines, including PoolFormer and related implementations

### Dependency

Install the basic dependencies with:

~~~bash
pip install torch torchvision
pip install timm
~~~


## Dataset, Training, and Evaluation

The training and evaluation scripts assume standard ImageNet-style directory layouts.

Example dataset paths used in this repository:

- ImageNet-1K: `/home/datasets/imagenet1k/`
- ImageNet-A: `/home/datasets/imagenet-a/`
- ImageNet-R: `/home/datasets/imagenet-r/`
- ImageNet-C: `/home/datasets/imagenet-c/`
- ImageNet-Sketch: `/home/datasets/imagenet-sk/`

Please replace these paths with your own local dataset paths.

### 1. Train GeoFormer

~~~bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
train_geometaformer.py \
  --data /home/datasets/imagenet1k/ \
  --model geometaformer_s12_k5 \
  --output ./output/geometaformer_s12_k5 \
  --experiment scratch_run01 \
> geometaformer_s12_k5.out 2>&1 &
~~~

### 2. Train MetaFormer baseline

~~~bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
train_metaformer.py \
  --data /home/datasets/imagenet1k/ \
  --model identityformer_s12 \
  --output ./output/identityformer_s12 \
  --experiment scratch_run01 \
> identityformer_s12.out 2>&1 &
~~~

### Robustness Evaluation

~~~bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
  --nproc_per_node=8 \
  --master_port=29611 \
  eval_robust.py \
  --model geometaformer_s12_k5 \
  --checkpoint /home/codes/pytorch-image-models/output/geometaformer_s12_k5/scratch_run01/model_best.pth.tar \
  --imagenet_a /home/datasets/imagenet-a/ \
  --imagenet_r /home/datasets/imagenet-r/ \
  --imagenet_c /home/datasets/imagenet-c/ \
  --imagenet_sk /home/datasets/imagenet-sk/ \
  --class_map_json ./wnid_to_idx_1k.json \
  --output_json geometaformer_s12_k5_robust.json
~~~

### Notes

- `train_geometaformer.py` is used for GeoFormer variants.
- `train_metaformer.py` is used for MetaFormer-style baselines.
- Please modify `CUDA_VISIBLE_DEVICES`, dataset paths, output paths, and model names according to your environment.
- Additional arguments supported by the underlying `timm` training pipeline may also be passed from the command line.
- The robustness evaluation script saves results into the JSON file specified by `--output_json`.

## Note

This work is currently under peer review, and a finalized citation entry is therefore not yet available. 

```bibtex
@article{qi_structured_invariance,
  title={Structured Invariance for Vision Backbones: A Geometric Route to MetaFormer},
  author={Qi, Shuren and Zhang, Yushu and Fang, Yuming and Cao, Xiaochun and Fan, Fenglei},
  journal={Preprint},
  year={2026}
}


For access to the manuscript and other related materials, including pretrained weights, please contact:

**Shuren Qi**  
Homepage: https://shurenqi.github.io/  
Email: shurenqi@cityu.edu.hk  

This project is built upon the excellent `timm` codebase. We sincerely thank the authors and contributors of `timm` for open-sourcing the training and model infrastructure.
