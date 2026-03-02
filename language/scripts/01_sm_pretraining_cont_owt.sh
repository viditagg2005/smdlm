#!/bin/bash

SEED=1
SAVEDIR="outputs/sm_mdlm_pretraining_cont_${SEED}"
CHECKPOINT_PATH=/u/her/003_results/08_diffusion/checkpoints/mdlm.ckpt
DATA_CACHE_DIR='.' # set this to your cache dir

mkdir $SAVEDIR

python -u -m main \
  trainer.max_steps=100_000 \
  data.cache_dir=$DATA_CACHE_DIR \
  loader.batch_size=32 \
  loader.eval_batch_size=32 \
  seed=$SEED \
  wandb.name="mdlm-sm-owt-ckpt-lr_01-sm_p-08-topk_3-seed${SEED}" \
  model=small \
  data=openwebtext-split \
  eval.compute_generative_perplexity=False \
  algo=mdlm_sm \
  algo.tran_head.mixinputs_k=3 \
  sampling.predictor=sm \
  optim.tran_head_lr=0.01 \
  optim.sm_prob=0.8 \
  strategy.find_unused_parameters=True\
  training.finetune_path=$CHECKPOINT_PATH \
  ++hydra.run.dir=$SAVEDIR 