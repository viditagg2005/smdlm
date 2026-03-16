#!/bin/bash
set -e

##############################################
# SM-MDLM SLERP Preliminary Test
# Single A100 GPU | WandB Disabled
##############################################

SEED=1
SAVEDIR="outputs/sm_mdlm_slerp_prelim_${SEED}"
DATA_CACHE_DIR='.'   # <-- set this to your data cache directory

# Disable WandB for prelim test
export WANDB_MODE=disabled

echo "============================================"
echo "  Step 1: Setup (fetch missing files + patch)"
echo "============================================"
cd "$(dirname "$0")/.."   # cd into language/
bash setup.sh

echo ""
echo "============================================"
echo "  Step 2: Install dependencies"
echo "============================================"
pip install -r requirements.txt
pip install flash_attn==2.7.4.post1 --no-build-isolation

echo ""
echo "============================================"
echo "  Step 3: Create output directory"
echo "============================================"
mkdir -p "$SAVEDIR"

echo ""
echo "============================================"
echo "  Step 4: Launch SLERP training"
echo "============================================"
python3 -u -m main \
  trainer.max_steps=100_000 \
  trainer.devices=1 \
  data.cache_dir=$DATA_CACHE_DIR \
  loader.batch_size=16 \
  loader.eval_batch_size=16 \
  loader.num_workers=8 \
  seed=$SEED \
  wandb.name="mdlm-sm-slerp-prelim-seed${SEED}" \
  model=small \
  algo=mdlm_sm \
  algo.tran_head.mixinputs_k=3 \
  algo.tran_head.interpolation=spherical \
  sampling.predictor=sm \
  optim.tran_head_lr=0.01 \
  optim.sm_prob=0.8 \
  trainer.val_check_interval=2000 \
  strategy=single_gpu \
  ++hydra.run.dir=$SAVEDIR

echo ""
echo "============================================"
echo "  Training complete!"
echo "  Checkpoints saved to: $SAVEDIR/checkpoints/"
echo "============================================"
