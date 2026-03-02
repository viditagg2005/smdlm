import torch
from transformers import DefaultDataCollator, DataCollatorWithPadding
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import torch
from utils import seeded_rand, seeded_randint, get_batch_seed

@dataclass
class CollatorConfig:
    """
    Configuration for the data collator.
    """
    mask_token_id: Optional[int] = None
    softmasking: bool = False
    softmasking_prob: float = 0.5
    min_prob: float = 0.2
    max_prob: float = 0.8

class dLLMDataCollator(DefaultDataCollator):
    """
    Batch-time padding and noising/masking
    """
    def __init__(self, tokenizer, cfg: CollatorConfig):
        super().__init__()
        self.tokenizer = tokenizer
        self.mask_token_id = tokenizer.mask_token_id if cfg.mask_token_id is None else cfg.mask_token_id
        if self.mask_token_id is None:
            raise ValueError("mask_token_id must be set (tokenizer.mask_token_id is None).")
        self.softmasking = cfg.softmasking # used in trainer
        self.softmasking_prob = cfg.softmasking_prob
        self.min_prob = cfg.min_prob
        self.max_prob = cfg.max_prob

    def add_at_least_one_mask(self, labels, input_ids, prompt_lengths, mask_indices, batch_seed=None):
        """
        Ensure that each row has at least one masked token.
        If a row has no masked tokens, randomly select one token (not in the prompt) to mask.
        """
        rows = (labels == -100).all(dim=1)
        if rows.any():
            print(f"Warning: {rows.sum().item()} rows have all labels as -100. Fixing by replacing one label per row.")
            idx = rows.nonzero(as_tuple=False).squeeze(1)
            low = int(prompt_lengths[idx][0].item())
            L = labels.size(1)
            
            j = seeded_randint(L-low, size=(1,), device=labels.device, seed=batch_seed) + low
            j = int(j.item())

            labels[idx, j] =input_ids[idx, j]
            input_ids[idx, j] = self.mask_token_id
            mask_indices[idx, j] = True
        return labels, input_ids, mask_indices
    
    def get_p_mask(self, batch, device, eps=1e-3, batch_seed=None):
        """
        Get the masking probability p_mask and the corresponding noise scale dsigma = 1/t
        """
        if "t" in batch:
            t = batch["t"].float()
            if t.ndim > 0: t = t.mean()
            t = t.clamp_min(eps)
        else:
            t = ((self.max_prob - self.min_prob) * seeded_rand((), device, seed=batch_seed) + self.min_prob).clamp_min(eps)   # Uniform[0.2,0.8]

        p_mask = (1 - eps) * t + eps
        dsigma = 1 / t
        return p_mask, dsigma

    def forward_process(self, batch, batch_seed=None, eps=1e-3):
        """
        Apply the forward noising/masking process to the input_ids in the batch.
        """
        input_ids = batch["input_ids"]
        prompt_lengths = batch["prompt_lengths"].long()
        device = input_ids.device
        B, N = input_ids.shape

        p_mask, dsigma = self.get_p_mask(batch, device, eps, batch_seed=batch_seed)

        mask_indices = seeded_rand((B, N), device, seed=batch_seed) < p_mask

        # Do not add noise to the prompt
        pos = torch.arange(N, device=device).unsqueeze(0)
        prompt_mask = pos < prompt_lengths.unsqueeze(1)
        mask_indices = mask_indices & (~prompt_mask)

        noisy_batch = input_ids.masked_fill(mask_indices, int(self.mask_token_id))
        labels = input_ids.clone().masked_fill(~mask_indices, -100)

        return noisy_batch, labels, mask_indices, p_mask, dsigma
    
    def prepad_input_ids(self, input_ids, pad_length, pad_token_id):
        """
        Pads the batch and adds a random number (0-max_extra) of pad tokens to the end of each sequence.
        """
        padded = torch.full((pad_length,), pad_token_id, dtype=input_ids.dtype, device=input_ids.device)
        padded[:input_ids.size(0)] = input_ids
        return padded

    def get_pad_length_for_batch(self, features, batch_seed=None):
        """
        Determine the padding length for the batch.
        Adds a random extra length (0-50) to the max input length in the batch
        """
        max_input_length_in_batch = max(len(f["input_ids"]) for f in features)

        extra_len = seeded_randint(50, size=(1,), device=features[0]["input_ids"].device, seed=batch_seed)
        max_len = max_input_length_in_batch + int(extra_len.item())

        # Make it a multiple of 8 for efficiency
        if max_len % 8 != 0:
            max_len += 8 - (max_len % 8)

        return max_len
        
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate and process a batch of features.
        """
        batch_seed = get_batch_seed(features)
        
        pad_length = self.get_pad_length_for_batch(features, batch_seed)
        for f in features:
            f["input_ids"] = self.prepad_input_ids(f["input_ids"], pad_length, self.tokenizer.pad_token_id)

        batch = super().__call__(features)

        # print("ORIG INPUTS ============", self.tokenizer.decode(batch["input_ids"][0], skip_special_tokens=False))

        noisy_batch, labels, mask_indices, p_mask, dsigma = self.forward_process(batch, batch_seed)
        labels, noisy_batch, mask_indices = self.add_at_least_one_mask(
            labels, noisy_batch, batch["prompt_lengths"].long(), mask_indices, batch_seed)

        batch.update({
            "input_ids": noisy_batch.long(),
            "labels": labels.long(),
            "mask_idx": mask_indices,      # optional, handy for debugging
            "p_mask": p_mask,              # scalar
            "dsigma": dsigma,              # scalar weighting = 1/t
        })

        # print("p_mask:", p_mask.item(), "dsigma:", dsigma.item())
        # print("MASKED INPUTS ============", self.tokenizer.decode(batch["input_ids"][0], skip_special_tokens=False))
        return batch