#!/bin/bash

# This script runs the evaluation for the kitchen-mixed-v0 environment
# with different sample timesteps.

# List of sample timesteps to evaluate
sample_timesteps_list=(100 75 60 50 30 25 20 15 12 10 6)

# Environment to use
env_name="kitchen-partial-v0"
checkpoint_dir="checkpoints/dtamp_kitchen-partial-v0"

for sample_steps in "${sample_timesteps_list[@]}"
do
    echo "Evaluating with sample_timesteps = $sample_steps"
    
    # Run the evaluation script with the current sample_timesteps
    OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=1 python scripts/d4rl/evaluate_dtamp.py \
        --env "$env_name" \
        --checkpoint_dir "$checkpoint_dir" \
        --config_path "config/d4rl/kitchen.yml" \
        --override_config "sample_timesteps=$sample_steps" \
        > "logs/evaluate_${env_name}_sample_steps_${sample_steps}.log" 2>&1 &

    # Wait for the process to finish before starting the next one
    wait
done

echo "All evaluations are done."
