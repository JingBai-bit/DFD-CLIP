#!/bin/bash

CUDA_VISIBLE_DEVICES=5 python main.py \
--dataset 'YawDD' \
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
--class-number "binary" \
--class-token-position "end" \
--class-specific-contexts 'True' \
--text-type 'class_descriptor' \
--seed 1 \
--temporal-layers 1 \
--num-segments 16 \
--dbl-m0 0.1 \
--dbl-lam 1.0 \
--dbl-m-max 0.5 \
--loss-fgw-weight 1.0 \
--fgw-beta 0.999 \
--fgw-lambda-F 1.0 \
--fgw-lambda-B 1.0 \
--fgw-tau 0.5 \
