import time
import torch
from transformers import AutoTokenizer, AutoModel, TrainingArguments, HfArgumentParser
from peft import LoraConfig, get_peft_model, TaskType
import os
from configs.args import DatasetArgs, RunArgs, LoraArgs, CustomTrainingArguments, TransparencyArgs
import sys
from components.preprocessor import load_data
from components.data_collator import CollatorConfig, dLLMDataCollator
from components.transparency_head import attach_transparency, require_grad_for_th
from components.trainer import dLLMTrainer
import random
import numpy as np

def init_seed(seed):
    """
    Initialize random seeds for reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Model loading
def load_model(run_args: RunArgs, trans_args: TransparencyArgs):
    """
    Load the pre-trained model and optionally attach the transparency head.
    """
    # Load model
    model = AutoModel.from_pretrained(
        run_args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    for p in model.parameters(): 
        p.requires_grad = False

    if run_args.softmasking:
        attach_transparency(model, trans_args)

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    print(model, flush=True)
    return model

def apply_lora(model, lora_args: LoraArgs, no_adapter: bool = False):
    """
    Apply LoRA adapters to the model if not disabled.
    """

    if no_adapter:
        return model

    # LoRA configuration
    print("Lora config:", lora_args, flush=True)
    cfg = LoraConfig(**vars(lora_args), 
                     task_type=TaskType.CAUSAL_LM)

    # Applying LoRA model
    t0 = time.time()
    model = get_peft_model(model, cfg)

    # if this is different across runs, we know the lora is initialized differently
    print("LoRA param checksum:", sum(p.sum().item() for n,p in model.named_parameters() if "lora_" in n))

    require_grad_for_th(model)
        
    model.print_trainable_parameters()   # should show a non-zero count
    print(f"[PEFT] wrapping took {time.time()-t0:.2f}s", flush=True)

    return model

# Training setup
def train_model(run_args, training_args, dataset_args, tokenizer, model, train_dataset, eval_dataset):
    """
    Set up the data collator and trainer, then start training.
    """
    softmasking = run_args.softmasking
    data_collator = dLLMDataCollator(
        tokenizer=tokenizer, 
        cfg=CollatorConfig(
            mask_token_id=tokenizer.mask_token_id,
            softmasking=softmasking,
            softmasking_prob=training_args.softmasking_prob,
            min_prob=dataset_args.min_prob,
            max_prob=dataset_args.max_prob,
        )
    )
    trainer = dLLMTrainer(
        model=model,
        args=training_args,
        loss_calc=training_args.loss_calc,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    trainable = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
    total_trainable = sum(k for _, k in trainable)
    print("Trainable params:", total_trainable)
    print([n for n,_ in trainable if "transparency" in n], flush=True)

    if run_args.no_adapter:
        # We just evaluate 100 time
        for _ in range(100):
            trainer.evaluate()
    else:
        # Start training
        if training_args.checkpoint_timestamp is not None:
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()

def main():
    """
    Main function to parse arguments, load model and data, and start training.
    """

    # Parse specific config file
    parser_run_and_dataset = HfArgumentParser((RunArgs, DatasetArgs))
    run_args, dataset_args = parser_run_and_dataset.parse_json_file(os.path.abspath(sys.argv[2]))

    # Parse common training arguments
    parser_training = HfArgumentParser((LoraArgs, CustomTrainingArguments, TransparencyArgs))
    lora_args, training_args, trans_args = parser_training.parse_json_file(os.path.abspath(sys.argv[1]))
    
    notes = "" if len(sys.argv) < 4 else sys.argv[3]

    # per-job output subdir
    if training_args.checkpoint_timestamp is not None:
        timestamp = training_args.checkpoint_timestamp
    else:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

    training_args.output_dir = os.path.join(training_args.output_dir, run_args.job_name, timestamp, notes)
    training_args.run_name = "___".join([run_args.job_name, notes, timestamp])
    
    # Seeding
    init_seed(training_args.seed)
    training_args.data_seed = training_args.seed

    tok = AutoTokenizer.from_pretrained(
        run_args.model_path, padding_side="right", trust_remote_code=True, use_fast=True)

    # Load dataset
    train_dataset, eval_dataset = load_data(dataset_args, tok, seed=training_args.data_seed)

    # Load model and tokenizer
    model = load_model(run_args, trans_args)
    model = apply_lora(model, lora_args, run_args.no_adapter)

    # Train the model
    print("Starting training...", flush=True)
    train_model(run_args, training_args, dataset_args, tok, model, train_dataset, eval_dataset)

if __name__ == "__main__":
    main()
