#
# Copyright 2026- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

# General imports
from tqdm import tqdm
import torch, hydra.utils
import torch.nn.functional as F

# Local imports
import trainer_base, utils
from trainer_base import sample_categorical
from transparency_head import TransparencyHead

class MDLM(trainer_base.AbsorbingState):
  def __init__(self, config, tokenizer):
    super().__init__(config, tokenizer)

  def _process_model_output(self, model_output, xt, sigma):
    del sigma
    model_output[:, :, self.mask_index] += self.neg_infinity
    
    # Normalize the model_output such that x.exp() is
    # a probability distribution over vocab_size.
    model_output = model_output - torch.logsumexp(
      model_output, dim=-1, keepdim=True)
    # Apply updates directly in the logits matrix.
    # For the logits of the unmasked tokens, set all values
    # to -infinity except for the indices corresponding to
    # the unmasked tokens.
    unmasked_indices = (xt != self.mask_index)
    model_output[unmasked_indices] = self.neg_infinity
    model_output[unmasked_indices, xt[unmasked_indices]] = 0
    return model_output

  def nll_per_token(self, log_x_theta, xt, x0, alpha_t,
                    dalpha_t, low_var=False):
    del xt
    log_p_theta = torch.gather(
      input=log_x_theta,
      dim=-1,
      index=x0[:, :, None]).squeeze(-1)
    return log_p_theta * dalpha_t / (1 - alpha_t)

  def _get_score(self, x, sigma):
    model_output = self.forward(x, sigma)
    # score(x, t) = p_t(y) / p_t(x)
    # => log score(x, t) = log p_t(y) - log p_t(x)
    
    # case 1: x = masked
    #   (i) y = unmasked
    #     log score(x, t) = log p_\theta(x)|_y + log k
    #     where k = exp(- sigma) / (1 - exp(- sigma))
    #   (ii) y = masked
    #     log score(x, t) = 0

    # case 2: x = unmasked
    #   (i) y != masked, y != x
    #     log score(x_i, t) = - inf
    #   (ii) y = x 
    #     log score(x_i, t) = 0
    #   (iii) y = masked token
    #     log score(x_i, t) = - log k
    #     where k = exp(- sigma) / (1 - exp(- sigma))
    
    log_k = - torch.log(torch.expm1(sigma)).squeeze(-1)
    assert log_k.ndim == 1
    
    masked_score = model_output + log_k[:, None, None]
    masked_score[:, :, self.mask_index] = 0

    unmasked_score = self.neg_infinity * torch.ones_like(
      model_output)
    unmasked_score = torch.scatter(
      unmasked_score,
      -1,
      x[..., None],
      torch.zeros_like(unmasked_score[..., :1]))
    unmasked_score[:, :, self.mask_index] = - (
      log_k[:, None] * torch.ones_like(x))
    
    masked_indices = (x == self.mask_index).to(
      model_output.dtype)[:, :, None]
    model_output = (
      masked_score * masked_indices
      + unmasked_score * (1 - masked_indices))
    return model_output.exp()


class MDLM_SM(MDLM):
    
  def __init__(self, config, tokenizer):
      super().__init__(config, tokenizer)
      # Initialize transparency head for soft feedback
      self.tran_head = TransparencyHead(mask_token_id = self.mask_index, 
                                trans_args = config.algo.tran_head)

  def _eval_mode(self):
      self.tran_head.eval()
      return super()._eval_mode()

  def _train_mode(self):
      self.tran_head.train()
      return super()._train_mode()

  def configure_optimizers(self):
      """
      Configures the optimizer with separate parameter groups for the main model
      and the TransparencyHead module.
      """
      # Separate the parameters into two groups
      special_lr_params = []
      main_params = []

      for name, param in self.named_parameters():
          # Check if the parameter belongs to the TransparencyHead module
          if name.startswith("tran_head."):
              special_lr_params.append(param)
          else:
              main_params.append(param)

      # Create the parameter groups with different learning rates
      param_groups = [
          {'params': main_params, 'lr': self.config.optim.lr},
          {'params': special_lr_params, 'lr': self.config.optim.tran_head_lr}
      ]

      # Instantiate the optimizer with the parameter groups
      optimizer = torch.optim.AdamW(
          param_groups,
          betas=(self.config.optim.beta1, self.config.optim.beta2),
          eps=self.config.optim.eps,
          weight_decay=self.config.optim.weight_decay
      )

      # Instantiate the learning rate scheduler
      scheduler = hydra.utils.instantiate(
          self.config.lr_scheduler, optimizer=optimizer
      )
      scheduler_dict = {
          'scheduler': scheduler,
          'interval': 'step',
          'monitor': 'val/loss',
          'name': 'trainer/lr'
      }

      return [optimizer], [scheduler_dict]

  def training_step(self, batch, batch_idx):
      # Log the computed transparency parameters (ω_s is the 'scale')
      self.log('transparency/omega_s', self.tran_head.scale.item(), on_step=True, on_epoch=False, sync_dist=True)
      self.log('transparency/centre', self.tran_head.centre.item(), on_step=True, on_epoch=False, sync_dist=True)
      self.log('transparency/steepness', self.tran_head.steepness.item(), on_step=True, on_epoch=False, sync_dist=True)
      self.log('transparency/temperature', self.tran_head.temperature.item(), on_step=True, on_epoch=False, sync_dist=True)

      # Log interpolation mode (0 = linear, 1 = spherical) for dashboard filtering
      is_spherical = 1.0 if self.tran_head.interpolation == "spherical" else 0.0
      self.log('transparency/is_spherical', is_spherical, on_step=True, on_epoch=False, sync_dist=True)

      # Call the parent training_step to compute and return the loss
      return super().training_step(batch, batch_idx)

  def forward(self, xt, sigma, log_p_x0=None):
    """
    Performs a forward pass with the option of using soft-masking.

    Args:
        xt: The input tensor of token ids.
        sigma: The noise level for the current timestep.
        log_p_x0: The model output from the previous step, used for feedback.

    Returns:
        The output logits from the model.
    """
    sigma_processed = self._process_sigma(sigma)

    with torch.cuda.amp.autocast(dtype=torch.float32):
      if log_p_x0 is not None:
          # Get embedding weight for potential SLERP usage
          embed_weight = self.backbone.vocab_embed.weight
          # If previous predictions are available, create a soft-masked input
          sm_out = self.tran_head(xt, log_p_x0, embed_weight=embed_weight)

          if self.tran_head.interpolation == "spherical":
              # SLERP path: sm_out is (B,T,D) dense embeddings
              # Pass directly as input_embeds, bypassing the embedding table
              model_output = self.backbone(
                  xt, sigma=sigma_processed, input_embeds=sm_out)
          else:
              # Linear path: sm_out is (indices, probs) tuple or (B,T,V) simplex
              model_output = self.backbone(sm_out, sigma=sigma_processed)
      else:
          # Standard forward pass if no previous prediction is available
          model_output = self.backbone(xt, sigma=sigma_processed)

    return self._process_model_output(model_output=model_output, xt=xt, sigma=sigma)

  
  def nll(self, x0, output_tokens, current_accumulation_step=None, train_mode=False):
    """
    Calculates the negative log-likelihood with find_unused_parameters=True.
    """
    del output_tokens
    t = self._sample_t(x0.shape[0], current_accumulation_step)
    assert t.shape[0] == x0.shape[0]
    if self.T > 0:
      t = (t * self.T).to(torch.int)
      t = t / self.T
      t += (1 / self.T)
    dalpha_t, alpha_t = self.noise(t)
    alpha_t = alpha_t.unsqueeze(-1)
    assert alpha_t.ndim == 2
    sigma = self._sigma_from_alphat(alpha_t)

    xt = self.q_xt(x0, alpha_t)

    use_soft_mask = not train_mode or (torch.rand(1).item() < self.config.optim.sm_prob)

    if use_soft_mask:
        # Pass 1: Get predictions for feedback. Block gradients
        log_x_theta_pass1 = self.forward(xt, sigma=sigma).detach()

        # Pass 2: Main pass that computes the gradients.
        log_x_theta = self.forward(xt, sigma=sigma, log_p_x0=log_x_theta_pass1)

        # --- SLERP diagnostics (every 100 steps, no grad) ---
        if (train_mode 
            and self.tran_head.interpolation == "spherical"
            and self.global_step % 100 == 0):
            self._log_slerp_angle_diagnostics(xt, log_x_theta_pass1)
    else:
        # --- Standard Path ---
        log_x_theta = self.forward(xt, sigma=sigma)

    utils.print_nans(log_x_theta, 'model_output')

    loss = self.nll_per_token(
          log_x_theta=log_x_theta,
          xt=xt,
          x0=x0,
          alpha_t=alpha_t,
          dalpha_t=dalpha_t,
          low_var=train_mode and self.loss_type == 'low_var')
    return loss
    
  def _log_slerp_angle_diagnostics(self, xt, logits_prelim):
    """
    Log SLERP angle diagnostics at masked positions.
    Tracks mean Ω (radians & degrees) and linear fallback fraction.
    """
    with torch.no_grad():
      mask_positions = (xt == self.mask_index)
      if not mask_positions.any():
          return

      embed_weight = self.backbone.vocab_embed.weight

      # Build per-position probability distribution
      if self.tran_head.transparency_alg == "mixinputs_with_topk":
          p = self.tran_head._get_topk_full_probs(logits_prelim)
      else:
          p = torch.softmax(logits_prelim, dim=-1)

      # Project into embedding space
      e_m = F.embedding(xt, embed_weight)                               # (B,T,D)
      e_pred = torch.matmul(p.to(embed_weight.dtype), embed_weight)     # (B,T,D)

      # L2 normalise
      eps = self.tran_head.epsilon
      e_hat_m    = e_m / (e_m.norm(dim=-1, keepdim=True) + eps)
      e_hat_pred = e_pred / (e_pred.norm(dim=-1, keepdim=True) + eps)

      # Angle at masked positions
      dot = (e_hat_m * e_hat_pred).sum(dim=-1).clamp(-1.0, 1.0)  # (B,T)
      omega = torch.acos(dot)                                      # (B,T)
      omega_masked = omega[mask_positions]
      sin_omega_masked = torch.sin(omega_masked)

      self.log('slerp/mean_omega_rad', omega_masked.mean().item(),
               on_step=True, on_epoch=False, sync_dist=True)
      self.log('slerp/mean_omega_deg', (omega_masked * 180.0 / 3.14159).mean().item(),
               on_step=True, on_epoch=False, sync_dist=True)
      self.log('slerp/linear_fallback_frac',
               (sin_omega_masked.abs() < eps).float().mean().item(),
               on_step=True, on_epoch=False, sync_dist=True)

  def generate_samples(self, num_samples, num_steps=None, eps=1e-5):
    """Generate samples from the model."""

    # Lightning auto-casting is not working in this method for some reason
    if num_steps is None:
      num_steps = self.config.sampling.steps
    x = self.prior_sample(num_samples, self.num_tokens)

    timesteps = torch.linspace(
      1, eps, num_steps + 1, device=self.device)
    dt = (1 - eps) / num_steps
    p_x0_cache = None

    p_x0_cache = None # Standard cache
    log_p_x0_cache_sm = None # Use log-probabilities for the SM cache

    confident_score = - torch.ones_like(x, device=self.device).to(torch.bfloat16) * torch.inf
    for i in tqdm(range(num_steps)):
        t = timesteps[i] * torch.ones(
        x.shape[0], 1, device=self.device)
        if self.sampler == 'ddpm_cache':
            log_p_x0_cache_sm, p_x0_cache, x_next, confident_score = self._ddpm_caching_update(
                x, t, dt, p_x0=p_x0_cache,log_p_x0_sm=log_p_x0_cache_sm, conf=confident_score)
            x = x_next
        else:
            x = self._analytic_update(x, t, dt)

    if self.config.sampling.noise_removal:
        min_t = timesteps[-1].item()
        t = min_t * torch.ones(x.shape[0], 1, device=self.device)
        if self.sampler == 'analytic':
            x = self._denoiser_update(x, t)
        else:
            unet_conditioning = self._sigma_from_alphat(self.noise(t)[1])
            # Use the final feedback pass for noise removal
            final_log_p_x0 = self.forward(x, unet_conditioning, log_p_x0=log_p_x0_cache_sm)
            x = final_log_p_x0.argmax(dim=-1)
    return x

  def _ddpm_caching_update(self, x, t, dt, p_x0=None,log_p_x0_sm=None,  conf=None):
      '''
      DDPM caching update borrowed and adapted from ReMDM. 
      '''
      assert self.config.noise.type == 'log-linear'
      sigma_t = self._sigma_from_alphat(self.noise(t)[1])
      if t.ndim > 1:
        t = t.squeeze(-1)
      assert t.ndim == 1
      move_chance_t = t[:, None, None]
      move_chance_s = (t - dt)[:, None, None]
      assert move_chance_t.ndim == 3, move_chance_t.shape
      if p_x0 is None:
        log_p_x0_sm = self.forward(x, sigma_t,log_p_x0_sm)
        p_x0 = log_p_x0_sm.exp()
        if self.config.sampling.p_nucleus < 1:
          sorted_probs, sorted_indices = torch.sort(p_x0, descending=True, dim=-1)
          cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
          top_p_mask = cumulative_probs <= self.config.sampling.p_nucleus
          top_p_mask[..., 0] = True
          nucleus_probs = sorted_probs * top_p_mask
          nucleus_probs /= nucleus_probs.sum(dim=-1, keepdim=True)
          p_x0 = torch.zeros_like(p_x0).scatter_(-1, sorted_indices, nucleus_probs)
  
      assert move_chance_t.ndim == p_x0.ndim

      if self.config.sampling.sampler == 'mdlm':
        q_xs = p_x0 * (move_chance_t - move_chance_s)
        q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
        _x = sample_categorical(q_xs)
        copy_flag = (x != self.mask_index).to(x.dtype)
        xs = copy_flag * x + (1 - copy_flag) * _x
      elif self.config.sampling.sampler == 'remdm-cap':
        alpha_t = (1 - move_chance_t)[0].item()
        alpha_s = (1 - move_chance_s)[0].item()
        if alpha_t > 0:
          sigma = min(self.config.sampling.eta, (1 - alpha_s) / alpha_t)
        else:
          sigma = self.config.sampling.eta
        q_xs = p_x0 * (1 - sigma)
        q_xs[..., self.mask_index] = sigma
        q_xs_2 = p_x0 * ((alpha_s - (1 - sigma) * alpha_t) / (1 - alpha_t))
        q_xs_2[..., self.mask_index] = (1 - alpha_s - sigma * alpha_t) / (1 - alpha_t)
        copy_flag = (x != self.mask_index).to(torch.bool)
        q_xs = torch.where(copy_flag.unsqueeze(-1), q_xs, q_xs_2)
        xs = sample_categorical(q_xs)
      elif self.config.sampling.sampler == 'remdm-loop':
        time = t[0].item()
        # compute alpha_t and alpha_s
        if time > self.config.sampling.t_on:
          move_chance_t = (1 - (1 - t) * self.config.sampling.alpha_on / (1 - self.config.sampling.t_on))[:, None, None]
          move_chance_s = (1 - (1 - t + dt) * self.config.sampling.alpha_on / (1 - self.config.sampling.t_on))[:, None, None]
        elif time <= self.config.sampling.t_off:
          move_chance_t = (t * (1 - self.config.sampling.alpha_on) / self.config.sampling.t_off)[:, None, None]
          move_chance_s = ((t - dt) * (1 - self.config.sampling.alpha_on) / self.config.sampling.t_off)[:, None, None]
        else:
          move_chance_t, move_chance_s = None, None
        # use MDLM
        if time > self.config.sampling.t_on or time <= self.config.sampling.t_off:
          q_xs = p_x0 * (move_chance_t - move_chance_s)
          q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
          _x = sample_categorical(q_xs)
          copy_flag = (x != self.mask_index).to(x.dtype)
          xs = copy_flag * x + (1 - copy_flag) * _x
        else: # use ReMDM
          sigma = self.config.sampling.eta
          q_xs = p_x0 * (1 - sigma)
          q_xs[..., self.mask_index] = sigma
          q_xs_2 = p_x0 * ((self.config.sampling.alpha_on - (1 - sigma) * self.config.sampling.alpha_on) / (1 - self.config.sampling.alpha_on))
          q_xs_2[..., self.mask_index] = (1 - self.config.sampling.alpha_on - self.config.sampling.alpha_on * sigma) / (1 - self.config.sampling.alpha_on)
          copy_flag = (x != self.mask_index).to(torch.bool)
          q_xs = torch.where(copy_flag.unsqueeze(-1), q_xs, q_xs_2)
          xs = sample_categorical(q_xs)
      
      if torch.allclose(xs, x) and not self.time_conditioning:
        p_x0_cache = p_x0
      else:
        p_x0_cache = None

      return log_p_x0_sm, p_x0_cache, xs, conf
