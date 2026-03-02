# Coding with Soft-Masked Diffusion Language Models

## 🔧 Requirements 

### Hardware

Running inference on the model requires a GPU with at least 20GB memory. We found that the training process requires at least 40GB memory. We run everything on one A100 with 40GB.

### Installing Dependencies

As mentioned by DreamLM: Dream-7B models are based on the [Huggingface `transformers`](https://github.com/huggingface/transformers) library. You should first install transformers by `pip install transformers>=4.46.2` and `torch>=2.5.1` as Dream uses the [SdpaAttention](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) built in torch. Other versions of transformers and torch are not been fully tested. 

For the SM finetuning, the only main requirement we add is `peft` (we use `peft==0.15.2`). The other requirements used for our setup are given in `requirements.txt`.

The easiest way to do this would be:
```
cd Dream-SM/
pip install -r requirements.txt
```

### Huggingface Connection

An account on huggingface is required to run these experiments. The first step is to update the code to your huggingface account. This can be done by changing any occurence of `<YOUR_HF_USERNAME>` to your huggingface username. We also assume that the reader has the huggingface CLI. More information on installing this can be found at https://huggingface.co/docs/huggingface_hub/en/guides/cli.

Test your setup by login into HF. 
``` bash
huggingface-cli login
```

### Applying patches and install libraries: 
Run the followin script to apply patches to the Dream models as well as the evaluation framework. The script will upload the Dream models to your HF name space. 
``` bash
bash setup.sh
```


## 🏋 Fine-tuning  

For finetuning `Dream-v0-Instruct-7B-SM`, run
``` bash
cd train
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=True

dataset="mix"
experiment="softmasking" # "no_softmasking"
seed=1 # 2, 3, 4, 5
notes=""

python train.py "./configs/base_configs/config_$seed.json" "./configs/datasets_$experiment/config_$dataset.json" "$notes"
```

For finetuning the coder model `Dream-v0-Coder-Instruct-7B-SM`, run
``` bash
cd train
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=True

dataset="coding"
experiment="softmasking" # "no_softmasking"
seed=1 # 2, 3, 4, 5
notes=""

python train.py "./configs/base_configs/config_$seed.json" "./configs/datasets_$experiment/config_$dataset.json" "$notes"
```
The dataset also specifies the base model to use. Refer to `configs/datasets_* for the exact setup.

### PEFT Adaptor Files

The output of the training is PEFT adaptor checkpoints with an additional `transparency_config.json`. This folder must be on huggingface to perform inference. To add the PEFT adaptor to your HF, first cd into your PEFT folder:
```
cd peft_adaptor_1/softmasking---mix/<timestamp/notes/checkpoint>
hf upload <YOUR_HF_NAME/><DESIRED_MODEL_NAME> .
```
An example `transparency_config.json` is as follows:
```json
{
    "transparency_alg": "mixinputs_with_topk",
    "transparency_scale": 0.25,
    "transparency_centre": -1,
    "transparency_steepness": 5,
    "mixinputs_k": 1,
    "mixture_temp": 1.0,
    "transparency_scheduling": "none"
}
```
This is necessary to have before running evaluation. The diffusion generate function will simply take these parameters as input. They vary based on learning process of the PEFT adaptor hence being included with the PEFT adaptor files.

## 📊 Evaluation

Once the Dream models have been finetuned and the soft-masking PEFT adaptors are available on your HF, you can run the evaluation. Make sure to put the path to the HF PEFT adaptor as the `model`.

Then, you can go to the `eval_instruct` directory and run the bash scripts:
```
cd eval_instruct
bash eval.sh
```
In eval.sh, there are various `model` parameters. Make sure `model` is set to the desired path (either the PEFT adaptor or the base model). There is also a flag that determines whether or not to add `softmasking`.

## General Notes 

### The Diffusion Generate Function

Refer to the Dream-7B github repo (https://github.com/DreamLM/Dream) for demos on how to use Dream-7B. Typically,, the generation process happens with the `diffusion_generate` function. An example call is given below:
```python
output = model.diffusion_generate(
    input_ids,
    attention_mask=attention_mask,
    max_new_tokens=512,
    steps=512,
    temperature=0.1,
    top_p=0.90,
    alg="entropy",
    alg_temp=0.,
)
generations = [
    tokenizer.decode(g[len(p) :].tolist())
    for p, g in zip(input_ids, output.sequences)
]

print(generations[0].split(tokenizer.eos_token)[0])
```

The main modification with SM mixing happens in this function on the HF backend. However, in order to perform SM, extra parameters must be added to `diffusion_generate`. In `./eval_instruct/lm_eval/models/diffllm.py`, we add an example of this:
```python
generation_ids = self.model.diffusion_generate(
    prompt_ids,
    attention_mask=attn_mask,
    max_new_tokens=self.max_new_tokens,
    output_history=False,
    return_dict_in_generate=True,
    steps=self.diffusion_steps,
    temperature=self.temperature,
    top_p=self.top_p,
    top_k=self.top_k,
    alg=self.alg,
    alg_temp=self.alg_temp,
    **th_kwargs
)
```
An example `th_kwargs` is:
```json
{
    "transparency_alg": "mixinputs_with_topk",
    "transparency_scale": 0.25,
    "transparency_centre": -1,
    "transparency_steepness": 5,
    "mixinputs_k": 1,
    "mixture_temp": 1.0,
    "transparency_scheduling": "none"
}
```
This `th_kwargs` is designed to be pulled from a `transparency_config.json` file included with the finetuned **PEFT adaptor. However, one can also just specify the parameters before calling diffusion generate.

### On the HF Backend

The `diffusion_generate` function is in the `generation_utils.py`. We add the `softmasking_utils.py` as the utility file. Behind the scenes, SM is patches this function added before calling the model. Rather than calling the model with `input_ids`, we create the desired embeddings and call the model with `inputs_embeds`. Specifically, the primary patch is:
```python
if sm_args.sm_alg != "none":
    p_sm = get_mixing_factors_for_softmasking(
        x, 
        logits, 
        mask_token_id, 
        max_gen_length, 
        sm_args
    )
    inputs_embeds = torch.matmul(p_sm, embed_weights)  # (B,T,D)
```

## Parameters of SM

SM requires a few additional parameters to `diffusion_generate` this effects the feedback that is given during the generation process:
- `transparency_alg`: (`none`, `mixinputs_with_topk`, or `mixinputs_with_temp`) Specifies which type of feedback to use. Top-k is the top-k tokens, and temp involves a learnable temperature for the soft-masking function.
- `transparency_scale`: The max scaling factor for the SM feedback sigmoid function.
- `transparency_centre`: The centre (or offset) for the SM feedback sigmoid function.
- `transparency_steepness`: The steepness parameter for the SM feedback sigmoid function.
- `mixinputs_k`: The number of tokens to include in feedback (only if `transparency_alg` is `mixinputs_with_topk`)
- `mixture_temp`: The softmaxing temperature for feedback (only if `transparency_alg` is `mixinputs_with_temp`)
- `transparency_scheduling`: Specifies the scheduling of SM during the decoding process (`none`, `linear`, or `stepwise`). Results are shown in ablations.

## Training

The training process uses the transformers [Trainer](https://huggingface.co/docs/transformers/en/main_classes/trainer) class extensively. All the configs we use are given in `./train/configs`. Here, we have `base_configs` which specify training and DoRA paramters. We also have `dataset` configs which specify configs for the training data. The model to finetune is also specified here because they are intertwined. 
The training procedure is quite simple:
- First we load the configs.
- Then we load the base model and `attach_transparency` which essentially attaches a learnable module that houses the SM logic during training. This was the simplest way to make the SM parameters learnable. However, this transparency head is only used for training purposes and not used during inference. The SM during inference is handled in the `diffusion_generate` function (as mentioned above).
- We then preprocess the data with some basic filtering.
- The data collator handles the padding and masking.
- The training for SM is done in a two-step-process with the first pass generating the "intermediate context" (with no gradient), and the second pass using the intermediate context.
- During training we only perform softmasking with a fraction of `softmasking_prob` (specified in base config).


## Acknowlegement
This part of the repository was built on top of [Dream-7B](https://github.com/DreamLM/Dream).

## Citation 📚
If you use the work released here for your research, please consider citing our paper:
```
@inproceedings{
hersche_softmasking_2026,
title={Soft-Masked Diffusion Language Models},
author={Hersche, Michael and Moor-Smith, Samuel and Hofmann, Thomas and Rahimi, Abbas},
booktitle={The Fourteenth International Conference on Learning Representations (ICLR)},
year={2026},
url={https://openreview.net/forum?id=Gba02UMvrG}
}
```
