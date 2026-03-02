from typing import Any, Dict, List
import numpy as np
from functools import partial
from datasets import concatenate_datasets, load_dataset, DatasetDict

LOAD_FROM_CACHE = True
# Used for coding SFT dataset cause they have different roles
role_map = {
    "HUMAN": "user",
    "ASSISTANT": "assistant"
}

def to_train_val(ds, test_size=0.1, seed=42):
    """
    Convert a dataset to a DatasetDict with 'train' and 'validation' splits.
    If the dataset already has 'validation' or 'test' splits, use them.
    Otherwise, split the 'train' split into train and validation sets.
    """
    try:
        if "validation" in ds:
            return DatasetDict(train=ds["train"], validation=ds["validation"])
        if "test" in ds:
            return DatasetDict(train=ds["train"], validation=ds["test"])
        split = ds["train"].train_test_split(test_size=test_size, seed=seed)
        return DatasetDict(train=split["train"], validation=split["test"])
    except:
        # single split dataset
        split = ds.train_test_split(test_size=test_size, seed=seed)
        return DatasetDict(train=split["train"], validation=split["test"])
    
def concat_after_split(d1, d2, seed=1):
    """
    Concatenate two DatasetDicts after splitting them into train and validation sets.
    """
    train = concatenate_datasets([d1["train"], d2["train"]]).shuffle(seed=seed)
    val   = concatenate_datasets([d1["validation"], d2["validation"]]).shuffle(seed=seed)
    return DatasetDict(train=train, validation=val)

def get_prompt_length(prompt, tokenizer):
    """
    Get the length of the tokenized prompt.
    """
    prompt_ids = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
    return len(prompt_ids)

def get_full_tokenized(prompt, response, tokenizer):
    """
    Get the full tokenized input_ids for the prompt and response.
    """
    full_ids = tokenizer.apply_chat_template(prompt + response, tokenize=True)
    return full_ids

def filter_long_inputs(features, max_length=2048):
    """
    Filters out sequences that are longer than max_length.
    """
    return features.filter(lambda ex: len(ex["input_ids"]) <= max_length, desc="filter long sequences", load_from_cache_file=LOAD_FROM_CACHE)

def filter_short_responses(features, min_length=5):
    """
    Filters out sequences that are shorter than min_length.
    """
    return features.filter(lambda ex: ex["resp_lengths"] >= min_length, desc="filter short sequences", load_from_cache_file=LOAD_FROM_CACHE)

def filter_tool_calls(features):
    """
    Filters out sequences that contain tool calls.
    """
    tool_call_token = 151657
    tool_call_end_token = 151658
    return features.filter(lambda ex: \
                           tool_call_token not in ex["input_ids"] and 
                           tool_call_end_token not in ex["input_ids"], 
                           desc="filter 151657/151658", load_from_cache_file=LOAD_FROM_CACHE)

def normalize_roles(msgs):
    """
    Normalize roles in messages using role_map (primarily for coding-SFT dataset).
    """
    return [{"role": role_map.get(m["role"], m["role"]), "content": m["content"]} for m in msgs]

def to_chat_batched(batch: Dict[str, List[Any]], dataset_args, tokenizer, max_length=2048) -> Dict[str, List[str]]:
    full_toks, prompt_lens, resp_lens = [], [], []
    if dataset_args.is_messages:
        msgs_list = batch["messages"]  # list[list[{"role":...,"content":...}], ...]
        for msgs in msgs_list:

            msgs = normalize_roles(msgs)
            
            p = msgs[:-1]
            r = msgs[-1:]
            
            full_tok = get_full_tokenized(p, r, tokenizer)
            prompt_len = get_prompt_length(p, tokenizer)
            resp_len = len(full_tok) - prompt_len

            full_toks.append(full_tok)
            prompt_lens.append(prompt_len)
            resp_lens.append(resp_len)
    else:
        assert dataset_args.prompt_column and dataset_args.response_column, \
            "For non-messages datasets, prompt_column and response_column must be specified."
        prompts  = batch[dataset_args.prompt_column]
        answers  = batch[dataset_args.response_column]
        for q, a in zip(prompts, answers):
            p = [{"role": "user", "content": q}]
            r = [{"role": "assistant", "content": a}]
            
            full_tok = get_full_tokenized(p, r, tokenizer)
            prompt_len = get_prompt_length(p, tokenizer)
            resp_len = len(full_tok) - prompt_len
            
            full_toks.append(full_tok)
            prompt_lens.append(prompt_len)
            resp_lens.append(resp_len)

    return {"input_ids": full_toks, "prompt_lengths": prompt_lens, "resp_lengths": resp_lens}

def load_data(args, tokenizer, test_size=0.1, seed=42):
    """
    Load and preprocess datasets based on the provided arguments.
    Applies necessary filtering and tokenization.
    """

    if args.data_path == "mix":
        d1 = to_train_val(load_dataset("allenai/tulu-3-sft-mixture"), test_size=test_size, seed=seed)
        d2 = to_train_val(load_dataset("HuggingFaceTB/smoltalk", "all"), test_size=test_size, seed=seed)
        raw = concat_after_split(d1, d2, seed=seed)
    else:
        ds = load_dataset(args.data_path, args.data_subset) if args.data_subset else load_dataset(args.data_path)
        raw = to_train_val(ds, test_size=test_size, seed=seed)

    raw["train"] = raw["train"].shuffle(seed=seed).select(range(min(args.max_train_size, len(raw["train"]))))
    raw["validation"] = raw["validation"].shuffle(seed=seed).select(range(min(500, len(raw["validation"]))))

    raw = raw.map(
        partial(to_chat_batched, dataset_args=args, tokenizer=tokenizer, max_length=args.max_length),
        batched=True, 
        remove_columns=raw["train"].column_names,
        desc="to tokenized chat",
        load_from_cache_file=LOAD_FROM_CACHE
    )
    
    # filter long/short sequences and tool calls
    raw["train"] = filter_long_inputs(raw["train"])
    raw["validation"] = filter_long_inputs(raw["validation"])

    raw["train"] = filter_short_responses(raw["train"])
    raw["validation"] = filter_short_responses(raw["validation"])

    raw["train"] = filter_tool_calls(raw["train"])
    raw["validation"] = filter_tool_calls(raw["validation"])

    # create t vals for the validation set
    tvals = np.linspace(args.min_prob, args.max_prob, num=len(raw["validation"]), dtype=np.float32).tolist()
    # create seeds for the validation set so its the same masking/padding each time
    seeds_train = [i + seed for i in range(len(raw["train"]))]
    seeds_val = [i + seed for i in range(len(raw["validation"]))]

    raw["validation"] = raw["validation"].add_column("t", tvals)

    raw["train"] = raw["train"].add_column("seed", seeds_train)
    raw["validation"] = raw["validation"].add_column("seed", seeds_val)

    # count tokens
    train_prompt_tokens = int(np.sum(raw["train"]["prompt_lengths"]))
    train_resp_tokens   = int(np.sum(raw["train"]["resp_lengths"]))
    print(f"Train tokens: prompt {train_prompt_tokens}, response {train_resp_tokens}, total {train_prompt_tokens+train_resp_tokens}")

    # set format for PyTorch
    raw["train"].set_format(type='torch', columns=["input_ids", "prompt_lengths", "seed"])
    raw["validation"].set_format(type='torch', columns=["input_ids", "prompt_lengths", "t", "seed"])

    print("Train size:", len(raw["train"]))
    print("Eval  size:", len(raw["validation"]))
    return raw["train"], raw["validation"]