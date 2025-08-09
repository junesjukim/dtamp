#!/bin/bash

# This script runs the evaluation for the kitchen-mixed-v0 environment.

OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=1 python scripts/d4rl/evaluate_dtamp.py \
    --env kitchen-mixed-v0 \
    --checkpoint_dir checkpoints/dtamp_kitchen-mixed-v0 \
    > logs/evaluate_kitchen-mixed-v0.log 2>&1 &

OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=2 python scripts/d4rl/evaluate_dtamp.py \
    --env kitchen-partial-v0 \
    --checkpoint_dir checkpoints/dtamp_kitchen-partial-v0 \
    > logs/evaluate_kitchen-partial-v0.log 2>&1 &