import json
import os
import torch
from transformers import Trainer
from torch.optim import AdamW
from components.transparency_head import transparency_head, get_th_kwargs
from utils import seeded_rand, get_batch_seed_collated

class dLLMTrainer(Trainer):

    def __init__(self, *args, 
                 loss_calc="model_weighted", 
                 data_collator=None,
                 **kwargs):
        print("TRAINER ARGS ======", args)
        print("TRAINER KWARGS =======", kwargs)
        super().__init__(*args, data_collator=data_collator, **kwargs)
        self.loss_calc = loss_calc

    def create_optimizer(self):
        """
        Create an optimizer with different learning rates for the base model and transparency head.
        """
        if self.optimizer is not None:
            return self.optimizer
        
        trans_head = transparency_head(self.model)
        extras = list(trans_head.parameters()) if trans_head is not None else []
        print("Extras count:", sum(p.numel() for p in extras))
        print("Extras requires_grad:", [p.requires_grad for p in extras][:5])
        extra_ids = {id(p) for p in extras}

        print("Extra trainable parameters:", flush=True)
        print(extra_ids, flush=True)

        # put every trainable param EXCEPT the transparency head into base
        base = [p for p in self.model.parameters()
            if p.requires_grad and id(p) not in extra_ids]
    
        self.optimizer = AdamW(
            [{"params": base,   "lr": self.args.learning_rate, "weight_decay": 0.0},
             {"params": extras, "lr": 1e-2,                    "weight_decay": 0.0}]
        )
        return self.optimizer

    def _do_softmasking(self, batch_seed) -> bool:
        """This function decides whether to do softmasking or not for this batch.
        The probability is given by self.data_collator.softmasking_prob.
        The randomness is seeded by the batch seed if evaluation, otherwise it is random.
        """

        if not getattr(self.data_collator, "softmasking", False):
            return False

        if self.model.training:
            softmasking_prob = seeded_rand((), self.model.device)
        else:
            softmasking_prob = seeded_rand((), self.model.device, seed=batch_seed)

        return softmasking_prob < self.data_collator.softmasking_prob
    
    def shift_logits(self, logits):
        """
        Shift logits to match the input_ids for the next token prediction.
        """
        return torch.cat([logits[:,:1], logits[:, :-1]], dim=1)

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        """
        Compute the loss, optionally using two-pass softmasking approach.
        The loss is weighted by dsigma = 1/t.
        """

        labels = inputs.pop("labels")
        dsigma = inputs.pop("dsigma")           # 0-dim tensor (batch-scalar = 1/t)
        batch_seed = get_batch_seed_collated(inputs)     # seed for this batch, or None
        p_mask = inputs.pop("p_mask", None)          # not used for weighting anymore

        x_t = inputs.get("input_ids", None)

        if self._do_softmasking(batch_seed):
            with torch.no_grad():
                logits_prelim = model(**inputs).logits
                logits_prelim = self.shift_logits(logits_prelim)

            # Store top-k logits and indices for the next step
            th_module = transparency_head(model)

            # if we are using embeddings, remove input_ids
            x_t = inputs.pop("input_ids")

            # Set the input_embeds via the transparency head forward pass
            p_sm = th_module(x_t, logits_prelim) # (B,T,V)

            # First get the input embeddings
            W = model.get_input_embeddings().weight  # (V,D)
            p_sm = p_sm.to(dtype=W.dtype, device=W.device) # (B,T,V)

            inputs["inputs_embeds"] = torch.matmul(p_sm, W)  # (B,T,D)

        if self.loss_calc == "model_weighted":
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            final_loss = dsigma * loss
        else: # Allow others to be added (ie. Context adaptive noise rescheduling)
            raise ValueError(f"Unknown loss_calc method: {self.loss_calc}")

        if self.state.global_step % self.args.logging_steps == 0:
            self.log_transparency(x_t, outputs.logits, p_mask)

        return (final_loss, outputs) if return_outputs else final_loss
    
    def log_transparency(self, x_t, logits, p_mask):
        """
        Log transparency head parameters if softmasking is used.
        """
        
        logits = self.shift_logits(logits)
        th_kwargs = get_th_kwargs(self.model)

        logs = {}
        if self.data_collator.softmasking and "transparency_scale" in th_kwargs.keys():
            # If softmasking, we log the transparency params
            with torch.no_grad():
                logs.update({
                    "transparency/scale":  th_kwargs["transparency_scale"],
                    "transparency/centre": th_kwargs["transparency_centre"],
                    "transparency/steep":  th_kwargs["transparency_steepness"],
                    "transparency/temperature": th_kwargs["mixture_temp"],
                })

            self.log(logs)

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False):
        """Save the th_kwargs instead of the module since inference just needs the kwargs."""
        # Let HF/PEFT do their normal thing (this will save LoRA adapters if present)
        super().save_model(output_dir, _internal_call=_internal_call)

        # Persist th_kwargs alongside the checkpoint in transparency_config.json
        try:
            th_kwargs = get_th_kwargs(self.model)
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "transparency_config.json"), "w", encoding="utf-8") as f:
                json.dump(th_kwargs, f, ensure_ascii=False, indent=2)
            self.log({"transparency/json_saved": 1})
        except Exception as e:
            print(f"Failed to save transparency_head.json: {e}", flush=True)
            try: self.log({"transparency/json_saved": 0})
            except Exception: pass