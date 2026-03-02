# 1. Download the Models (Hugging Face)
# Download the actual model weights and their associated Python execution files from the Dream-org repository on Hugging Face into a `models/` directory.

# Download Dream Instruct 7B
huggingface-cli download Dream-org/Dream-v0-Instruct-7B --local-dir models/Dream-v0-Instruct-7B-SM

# Download Dream Coder Instruct 7B
huggingface-cli download Dream-org/Dream-Coder-v0-Instruct-7B --local-dir models/Dream-Coder-v0-Instruct-7B-SM
```

# 2. Download the Evaluation Code (GitHub)

# Use ghclone to download only the eval_instruct folder directly from the DreamLM GitHub repository into your project root.

ghclone [https://github.com/DreamLM/Dream/tree/main/eval_instruct](https://github.com/DreamLM/Dream/tree/main/eval_instruct)

# 3. Apply the Patches

# We have two patches stored in the `patches/` directory:

# Patch the Instruct model files
cd models/Dream-v0-Instruct-7B-SM
patch -p1 < ../../patches/dream_sm.patch
cd ../..

# Patch the Coder-Instruct model files
cd models/Dream-Coder-v0-Instruct-7B-SM
patch -p1 < ../../patches/dream_sm.patch
cd ../..

# Patch the eval_instruct module
cd eval_instruct
patch -p1 < ../patches/eval_instruct.patch
cd ..

# 4. Upload the models to HF
export HF_USER=$(hf auth whoami | head -n 1)

# Upload the patched models
hf upload ${HF_USER}/Dream-v0-Instruct-7B-SM ./models/Dream-v0-Instruct-7B-SM
hf upload ${HF_USER}/Dream-Coder-v0-Instruct-7B-SM ./models/Dream-Coder-v0-Instruct-7B-SM

# 5. Setting up lm-evaluation-harness
# The evaluation is based on [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), so you should first install it with:
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
cd ..
# To install the our patched `lm_eval`:
cd eval_instruct
pip install -e .
cd ..

echo "Installation completed"