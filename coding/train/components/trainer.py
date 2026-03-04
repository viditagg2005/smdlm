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

            # First get the input embeddings weight matrix
            W = model.get_input_embeddings().weight  # (V,D)

            # Set the input_embeds via the transparency head forward pass
            sm_out = th_module(x_t, logits_prelim, embed_weight=W)

            if th_module.interpolation == "spherical":
                # SLERP path: sm_out is already (B,T,D) dense embeddings
                inputs["inputs_embeds"] = sm_out
            else:
                # Linear path: sm_out is (B,T,V) probability simplex
                p_sm = sm_out.to(dtype=W.dtype, device=W.device)
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
        Includes SLERP-specific diagnostics when using spherical interpolation.
        """
        
        logits = self.shift_logits(logits)
        th_kwargs = get_th_kwargs(self.model)

        logs = {}
        if self.data_collator.softmasking and "transparency_scale" in th_kwargs.keys():
            # If softmasking, we log the transparency params
            # ω_s is the critical scaling factor to monitor
            with torch.no_grad():
                logs.update({
                    "transparency/omega_s":  th_kwargs["transparency_scale"],
                    "transparency/centre": th_kwargs["transparency_centre"],
                    "transparency/steep":  th_kwargs["transparency_steepness"],
                    "transparency/temperature": th_kwargs["mixture_temp"],
                })

                # Log interpolation mode (0 = linear, 1 = spherical)
                interp_mode = th_kwargs.get("interpolation", "linear")
                logs["transparency/is_spherical"] = 1.0 if interp_mode == "spherical" else 0.0

                # SLERP-specific diagnostics: compute angle stats
                if interp_mode == "spherical" and x_t is not None:
                    th_module = transparency_head(self.model)
                    W = self.model.get_input_embeddings().weight
                    try:
                        self._log_slerp_diagnostics(logs, th_module, x_t, logits, W)
                    except Exception:
                        pass  # Don't crash training on diagnostic failures

            self.log(logs)

    def _log_slerp_diagnostics(self, logs, th_module, x_t, logits, W):
        """
        Compute and log SLERP angle diagnostics for masked positions.
        This helps detect:
          - Angle Ω collapsing to 0 (vectors too similar → SLERP degenerates)
          - Too many positions falling back to linear (sin(Ω) ≈ 0)
        """
        mask_positions = (x_t == th_module.mask_token_id)
        if not mask_positions.any():
            return

        # Get top-k probability distribution for masked positions
        if th_module.transparency_alg == "mixinputs_with_topk":
            p = th_module.get_only_topk_probs(logits, th_module.mixinputs_k)
        else:
            p = torch.softmax(logits, dim=-1)

        # Project into embedding space
        import torch.nn.functional as F
        e_m = F.embedding(x_t, W)                                  # (B, T, D)
        e_pred = torch.matmul(p.to(W.dtype), W)                    # (B, T, D)

        # L2 normalise
        eps = th_module.epsilon
        e_hat_m    = e_m / (e_m.norm(dim=-1, keepdim=True) + eps)
        e_hat_pred = e_pred / (e_pred.norm(dim=-1, keepdim=True) + eps)

        # Compute angle Ω only at masked positions
        dot = (e_hat_m * e_hat_pred).sum(dim=-1)                    # (B, T)
        dot = dot.clamp(-1.0, 1.0)
        omega = torch.acos(dot)                                     # (B, T)
        omega_masked = omega[mask_positions]                         # (N_masked,)
        sin_omega_masked = torch.sin(omega_masked)

        logs["slerp/mean_omega_rad"]       = omega_masked.mean().item()
        logs["slerp/mean_omega_deg"]       = (omega_masked * 180.0 / 3.14159).mean().item()
        logs["slerp/linear_fallback_frac"] = (sin_omega_masked.abs() < eps).float().mean().item()

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