# DTRT: Dynamically Tokenized Relation Transformer

Official implementation for **End-to-End Hyper-Relational Information Extraction for Engineering Diagrams via Dynamically Tokenized Relation Transformer**.

DTRT targets engineering diagram digitization. Instead of treating symbols, lines, and text as separate detection subtasks, it formulates diagram parsing as end-to-end scene/hyper-relational graph generation: entities are detected, component relations are predicted, and textual labels can be organized as qualifiers for entities or relations.

## Core Claims

- **End-to-end diagram parsing**: DTRT predicts diagram entities and relation triplets in one framework, reducing hand-built multi-stage parsing pipelines.
- **Hyper-relational knowledge graph output**: the model is designed for structured engineering knowledge extraction rather than only bounding-box localization.
- **Dynamic tokenization**: low-value visual tokens are pruned inside the Swin visual backbone to reduce computation on high-resolution diagrams.
- **Reconstruction-supervised pruning**: an optional reconstruction branch supervises token scorers, helping preserve thin lines, text, and other details that are easy to discard.
- **Engineering-diagram focus**: the code is configured for P&ID-style diagram parsing through PID2Graph and keeps the original Visual Genome/Open Images paths for compatibility.

The paper reports strong performance on P&IDs and electrical diagrams, including **94.84% R@1000 on P&IDs** and **92.52% R@200 on EDs**, with substantially reduced computation from dynamic tokenization. Exact reproduction requires the paper data splits, annotations, and checkpoint configuration.

## Repository Layout

```text
.
├── main.py                         # training/evaluation entrypoint
├── inference.py                    # single-image inference entrypoint
├── models/                         # DTRT/RelTR model, transformer, backbone, matcher
├── datasets/                       # COCO-style datasets, transforms, evaluation hooks
├── lib/                            # scene graph and Open Images evaluation utilities
├── tools/                          # PID2Graph preparation and training scripts
├── data/                           # dataset metadata/annotations, not committed
└── ckpt/                           # checkpoints, not committed
```

Large artifacts such as datasets, generated annotations, checkpoints, logs, and caches are intentionally ignored by `.gitignore`.

## Installation

Create an environment with PyTorch, torchvision, scipy, pycocotools, and matplotlib. A recent torchvision version is required for Swin backbones.

```bash
conda create -n dtrt python=3.10 -y
conda activate dtrt
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
conda install scipy matplotlib -y
pip install pycocotools
```

For the original Visual Genome/Open Images evaluation utilities, compile the box intersection extension:

```bash
cd lib/fpn
sh make.sh
cd ../..
```

## Data Preparation

### PID2Graph / P&ID

Place the PID2Graph raw files under:

```text
PID2Graph/Complete/
```

Then generate the local annotation files:

```bash
python tools/prepare_pid2graph.py
```

The generated structure is:

```text
data/pid2graph/
├── train.json
├── val.json
├── test.json
└── rel.json
```

Train and evaluate with:

```bash
--dataset pid2graph \
--img_folder PID2Graph/Complete/ \
--ann_path data/pid2graph/
```

### Visual Genome / Open Images

The original RelTR data paths are retained for compatibility. Follow `data/README.md` and use this structure:

```text
data/
├── vg/
│   ├── train.json
│   ├── val.json
│   ├── test.json
│   ├── rel.json
│   └── images/
└── oi/
    ├── train.json
    ├── val.json
    ├── test.json
    ├── rel.json
    └── images/
```

## Model Options

### Backbone

Swin is the default backbone:

```bash
--backbone swin_t
```

Supported torchvision backbones include:

- `swin_t`, `swin_s`, `swin_b`
- `swin_v2_t`, `swin_v2_s`, `swin_v2_b`
- `resnet50` and other original torchvision ResNet variants



### Dynamic Tokenization

Token pruning is enabled for Swin by default:

```bash
--swin_prune_keep_ratios 0.9,0.7,0.5 \
--swin_prune_stages 3,5,7
```

Use `--disable_swin_token_pruning` for a non-pruned Swin baseline.

### Reconstruction Branch

Enable reconstruction-supervised pruning with:

```bash
--enable_token_reconstruction \
--token_recon_loss_coef 1.0
```

By default, reconstruction loss is applied to pruned valid tokens only. Add `--token_recon_all_tokens` to supervise all valid tokens.

## Training

### P&ID Fine-Tuning

Run the PID2Graph fine-tuning baseline:

```bash
bash tools/finetune_pid2graph.sh
```

### DTRT with Dynamic Tokenization

Run Swin + token pruning:

```bash
bash tools/finetune_pid2graph_swin_prune.sh
```

Equivalent key options:

```bash
python main.py \
  --dataset pid2graph \
  --img_folder PID2Graph/Complete/ \
  --ann_path data/pid2graph/ \
  --backbone swin_t \
  --swin_img_size 224 \
  --swin_prune_keep_ratios 0.9,0.7,0.5 \
  --swin_prune_stages 3,5,7 \
  --num_classes 11 \
  --num_rel_classes 2 \
  --num_entities 800 \
  --num_triplets 900 \
  --batch_size 1 \
  --output_dir ckpt/pid2graph_swin_t_prune
```

### DTRT with Reconstruction-Supervised Pruning

Run Swin + token pruning + reconstruction loss:

```bash
bash tools/finetune_pid2graph_swin_prune_recon.sh
```

If your conda environment is not named `reltr`, set:

```bash
RELTR_CONDA_ENV=my_env bash tools/finetune_pid2graph_swin_prune_recon.sh
```

## Evaluation

PID2Graph / P&ID:

```bash
python main.py \
  --dataset pid2graph \
  --img_folder PID2Graph/Complete/ \
  --ann_path data/pid2graph/ \
  --backbone swin_t \
  --swin_img_size 224 \
  --num_classes 11 \
  --num_rel_classes 2 \
  --num_entities 800 \
  --num_triplets 900 \
  --sg_recall_ks 20,50,100,200 \
  --coco_max_dets 1,10,300 \
  --eval \
  --batch_size 1 \
  --resume ckpt/YOUR_CHECKPOINT.pth
```


```

## Inference

Run image inference with a trained checkpoint:

```bash
python inference.py --img_path demo/vg1.jpg --resume ckpt/YOUR_CHECKPOINT.pth
```

For engineering diagrams, use a checkpoint trained with matching class/relation definitions.



## Citation

If this code or paper helps your research, please cite DTRT:

```bibtex
@inproceedings{bai2026dtrt,
  title={End-to-End Hyper-Relational Information Extraction for Engineering Diagrams via Dynamically Tokenized Relation Transformer},
  author={Bai, Tianyou and Zhang, Yan-Ming and Zhang, Zixiang and Zhou, Jibin and Yin, Fei and Liu, Cheng-Lin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2026}
}
```

This project builds on RelTR:

```bibtex
@article{cong2023reltr,
  title={Reltr: Relation transformer for scene graph generation},
  author={Cong, Yuren and Yang, Michael Ying and Rosenhahn, Bodo},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2023},
  publisher={IEEE}
}
```
