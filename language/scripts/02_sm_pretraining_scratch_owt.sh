#!/bin/bash

SEED=1
SAVEDIR="outputs/sm_mdlm_pretraining_scratch_${SEED}"
CHECKPOINT_PATH=checkpoints/mdlm.ckpt
DATA_CACHE_DIR='.' # set this to your cache dir

python -u -m main \
  trainer.max_steps=1_000_000 \
  data.cache_dir=$DATA_CACHE_DIR \
  loader.batch_size=32 \
  loader.eval_batch_size=32 \
  loader.num_workers=8 \
  seed=$SEED \
  wandb.name="mdlm-sm-owt-scratch-lr_01-sm_p-08-topk_3-seed${SEED}" \
  model=small \
  algo=mdlm_sm \
  algo.tran_head.mixinputs_k=3 \
  sampling.predictor=sm \
  optim.tran_head_lr=0.01 \
  optim.sm_prob=0.8 \
  strategy.find_unused_parameters=True\
  ++hydra.run.dir=$SAVEDIR