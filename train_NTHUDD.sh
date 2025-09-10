#!/bin/bash

CUDA_VISIBLE_DEVICES=3 python main.py \
--dataset 'NTHUDDD' \
--workers 4 \
--epochs 50 \
--batch-size 24 \
--lr 1e-2 \
--lr-image-encoder 1e-5 \
--lr-prompt-learner 1e-3 \
--weight-decay 1e-4 \
--momentum 0.9 \
--print-freq 10 \
--milestones 30 40 \
--contexts-number 8 \
--class-token-position "end" \
--class-specific-contexts 'True' \
--text-type 'class_descriptor' \
--seed 1 \
--temporal-layers 1