#!/usr/bin/env bash
set -euo pipefail

RELTR_CONDA_ENV=${RELTR_CONDA_ENV:-reltr}

conda run -n "$RELTR_CONDA_ENV" python main.py \
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
  --sg_recall_ks 20,50,100,200 \
  --coco_max_dets 1,10,300 \
  --no_backbone_pretrained \
  --pretrained ckpt/checkpoint0149.pth \
  --batch_size 1 \
  --epochs 50 \
  --lr_drop 35 \
  --num_workers 0 \
  --output_dir ckpt/pid2graph_swin_t_prune
