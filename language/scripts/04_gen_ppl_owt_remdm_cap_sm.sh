#!/bin/bash

sampling_steps=${1:-128}
SEED=1
experiment="sm_mdlm_pretraining_cont_1"
checkpoint_name="5-100000"
checkpoint_path=./outputs/${experiment}/checkpoints/${checkpoint_name}.ckpt
p=0.9
eta=0.04
generated_seqs_path=./outputs/${experiment}/generations/remdm-cap_${checkpoint_name}_T-${sampling_steps}_eta-${eta}_topp-${p}.json
DATA_CACHE_DIR='.' # set this to your cache dir
mkdir ./outputs/${experiment}/generations

export HYDRA_FULL_ERROR=1

python -u -m main \
    data.cache_dir=$DATA_CACHE_DIR \
    mode=sample_eval \
    loader.batch_size=1 \
    loader.eval_batch_size=1 \
    eval.perplexity_batch_size=1 \
    algo=mdlm_sm \
    model.length=1024 \
    eval.checkpoint_path=${checkpoint_path} \
    +wandb.offline=true \
    hydra.run.dir="${PWD}" \
    sampling.steps=${sampling_steps} \
    seed=$SEED \
    sampling.num_sample_batches=5000 \
    sampling.generated_seqs_path=${generated_seqs_path} \
    sampling.p_nucleus=${p} \
    sampling.sampler="remdm-cap" \
    sampling.eta=${eta} 