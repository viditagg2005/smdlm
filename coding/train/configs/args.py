# args.py
from dataclasses import dataclass, field
from typing import List, Optional
from transformers import TrainingArguments

@dataclass
class RunArgs:
    job_name: str = "default_job"
    model_path: str = "<YOUR_HF_USERNAME>/Dream-v0-Instruct-7B-SM"
    softmasking: bool = True
    no_adapter: bool = False

@dataclass
class DatasetArgs:
    data_path: str = "mix"
    data_subset: str = None
    is_messages: bool = True  # Whether the dataset uses "messages" format
    prompt_column: str = None
    response_column: str = None
    max_length: int = 2048
    max_train_size: int = 300000

    min_prob: float = 0.2
    max_prob: float = 0.8

@dataclass
class TransparencyArgs:

    mixinputs_k: int = 3
    transparency_alg: str = "mixinputs_with_topk"  # Options: "mixinputs", "

    init_scale: float = 0.0
    init_centre: float = -0.75
    init_steep: float = 10/1.5

@dataclass
class LoraArgs:
    r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0 # 0.1
    bias: str = "none"
    use_dora: bool = True
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ]), # , "gate_proj", "up_proj", "down_proj"
    modules_to_save: List[str] = field(default_factory=lambda: ["transparency"]), # current implementation does not save this since we just use the transparency_config.json parameters during inference
    layers_to_transform: Optional[List[int]] = None, # list(range(5)),
    layers_pattern: Optional[str] = None, # "model.layers."

@dataclass
class CustomTrainingArguments(TrainingArguments):
    loss_calc: str = "model_weighted",  # Options: "diffullama", "model_unweighted", "model_weighted"
    checkpoint_timestamp: Optional[str] = None  # e.g. "2025-09-10_13-07-54"
    softmasking_prob: float = 0.8  # Probability of using softmasking in each batch
