#!/bin/bash
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=True

# First command-line argument is the dataset name; default to "mix"
# This also specifies the base model to use (Dream or Dream-Coder)
# Refer to configs/datasets_*/config_{dataset}.json for the exact setup
dataset="${1:-mix}"

runs=(
    "no_softmasking"
    "softmasking"
)

for run in "${runs[@]}"; do
    for seed in 1 2 3 4 5; do

        notes="" # Add any desired run notes here

        # Get the base config file based on the seed
        base_config="./configs/base_configs/config_$seed.json"

        timestamp=$(date +%Y%m%d_%H%M%S)
        echo "Submitting job job for experiment: $run and $dataset, at timestamp: $timestamp"

        python train.py "$base_config" "./configs/datasets_$run/config_$dataset.json" "$notes"
        
    done
done