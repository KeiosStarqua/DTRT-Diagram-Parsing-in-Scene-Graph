#!/usr/bin/env bash
set -euo pipefail

python tools/prepare_pid2graph.py

RELTR_CONDA_ENV=${RELTR_CONDA_ENV:-reltr}

conda run -n "$RELTR_CONDA_ENV" python main.py \
  --dataset pid2graph \
  --img_folder PID2Graph/Complete/ \
  --ann_path data/pid2graph/ \
  --num_classes 11 \
  --num_rel_classes 2 \
  --num_entities 800 \
  --num_triplets 900 \
  --sg_recall_ks 20,50,100,200,500,900 \
  --coco_max_dets 1,10,800 \
  --no_backbone_pretrained \
  --pretrained ckpt/checkpoint0149.pth \
  --batch_size 1 \
  --epochs 50 \
  --lr_drop 35 \
  --output_dir ckpt/pid2graph_finetune_q800_t900
