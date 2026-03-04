import torch
import torch.nn as nn
import torch.nn.functional as F

def softplus_inv_param(init) -> torch.Tensor:
    """
    Given a desired initial value `init` for a parameter that will be transformed. 
    This will ensure that the parameter is always positive.
    """
    return nn.Parameter(torch.log(torch.expm1(torch.tensor(init, dtype=torch.float32))))

class TransparencyHead(nn.Module):
    """
    A transparency head that performs softmasking for text generation models.
    We only use this during training to learn the parameters, not during inference.
    During inference, we simply add the parameters to the diffusion_generate function and SM is handled by the model during generation.
    """
    def __init__(self, mask_token_id, trans_args):
        super().__init__()
            
        self.mask_token_id = mask_token_id

        init_scale = getattr(trans_args, "init_scale", 0.0)
        init_centre = getattr(trans_args, "init_centre", -0.75)
        init_steep = getattr(trans_args, "init_steep", 10/1.5)
        init_temperature = getattr(trans_args, "init_temperature", 1.0)

        self.raw_scale = nn.Parameter(torch.logit(torch.tensor(init_scale), eps=1e-6)) # keeps in (0,1)
        self.raw_centre_neg = softplus_inv_param(-init_centre)
        self.raw_steep = softplus_inv_param(init_steep)
        self.raw_temperature = softplus_inv_param(init_temperature)

        self.mixinputs_k = getattr(trans_args, "mixinputs_k", 3) 
        self.transparency_alg = getattr(trans_args, "transparency_alg", "mixinputs_with_topk")
        self.interpolation = getattr(trans_args, "interpolation", "linear")

        self.epsilon = 1e-6

    @property
    def scale(self):
        return torch.sigmoid(self.raw_scale)

    @property
    def centre(self):
        return -F.softplus(self.raw_centre_neg) - self.epsilon

    @property
    def steepness(self):
        return F.softplus(self.raw_steep) + self.epsilon

    @property
    def temperature(self):
        return F.softplus(self.raw_temperature) + self.epsilon

    def forward(self, input_ids, logits_prelim, embed_weight=None):
        """
        Args:
            input_ids:      (B, T) token ids.
            logits_prelim:  (B, T, V) logits from the preliminary pass.
            embed_weight:   (V, D) embedding weight matrix. Required when
                            interpolation == "spherical"; ignored otherwise.
        Returns:
            If interpolation == "linear":  (B, T, V) probability simplex.
            If interpolation == "spherical": (B, T, D) dense embeddings.
        """

        # --- 1. Compute lambda (shared by both modes) ---
        temperature = self.temperature if self.transparency_alg == "mixinputs_with_temp" else 1.0
        neg_entropy, p = self.get_neg_entropy_and_probabilities(logits_prelim, temperature=temperature)

        mask_positions = (input_ids == self.mask_token_id)
        lambda_tensor = self.calculate_lambda_tensor(neg_entropy, mask_positions)  # (B,T,1)

        if self.transparency_alg == "mixinputs_with_topk":
            p = self.get_only_topk_probs(logits_prelim, self.mixinputs_k)

        # --- 2. Branch on interpolation mode ---
        if self.interpolation == "spherical":
            assert embed_weight is not None, \
                "embed_weight must be provided for spherical interpolation"
            return self._slerp_interpolate(input_ids, p, lambda_tensor, embed_weight)
        else:
            # Original linear path: returns (B,T,V) probability simplex
            xt_one_hot = F.one_hot(input_ids, num_classes=logits_prelim.shape[-1]).to(logits_prelim.dtype)
            p_out = (1 - lambda_tensor) * xt_one_hot + lambda_tensor * p
            return p_out

    def _slerp_interpolate(self, input_ids, p, lambda_tensor, embed_weight):
        """
        Spherical Linear Interpolation (SLERP) in the dense embedding space.

        Args:
            input_ids:    (B, T)   token ids (mask token at masked positions).
            p:            (B, T, V) probability distribution over vocab.
            lambda_tensor:(B, T, 1) confidence weight per position.
            embed_weight: (V, D)   embedding table weights.

        Returns:
            e_final:      (B, T, D) dense embeddings ready for the backbone.
        """
        # Step 1: Project into dense embedding space
        e_m = F.embedding(input_ids, embed_weight)           # (B, T, D)
        e_pred = torch.matmul(p.to(embed_weight.dtype), embed_weight)  # (B, T, D)

        # Step 2: L2 normalisation
        e_m_norm = e_m.norm(dim=-1, keepdim=True)            # (B, T, 1)
        e_hat_m    = e_m / (e_m_norm + self.epsilon)
        e_hat_pred = e_pred / (e_pred.norm(dim=-1, keepdim=True) + self.epsilon)

        # Step 3: Compute angle Ω between the two unit vectors
        dot = (e_hat_m * e_hat_pred).sum(dim=-1, keepdim=True)  # (B, T, 1)
        dot = dot.clamp(-1.0, 1.0)
        omega = torch.acos(dot)                                 # (B, T, 1)
        sin_omega = torch.sin(omega)                            # (B, T, 1)

        # Step 4: SLERP with linear fallback for small angles
        safe = sin_omega.abs() > self.epsilon

        # SLERP coefficients (only valid where safe == True)
        coeff_m    = torch.sin((1.0 - lambda_tensor) * omega) / (sin_omega + self.epsilon)
        coeff_pred = torch.sin(lambda_tensor * omega)          / (sin_omega + self.epsilon)

        # Linear fallback coefficients
        coeff_m_lin    = 1.0 - lambda_tensor
        coeff_pred_lin = lambda_tensor

        # Select per-position
        coeff_m    = torch.where(safe, coeff_m,    coeff_m_lin)
        coeff_pred = torch.where(safe, coeff_pred, coeff_pred_lin)

        e_slerp = coeff_m * e_hat_m + coeff_pred * e_hat_pred   # (B, T, D)

        # Step 5: Rescale by the original mask-token embedding magnitude
        e_final = e_slerp * e_m_norm

        return e_final

    def get_neg_entropy_and_probabilities(self, logits, temperature=1.0):
        """Get negative entropy and probabilities from logits"""
        epsilon = 1e-10
        p = torch.softmax(logits / temperature, dim=-1)   # (B,T,V)
        logp = torch.log(p + epsilon)
        neg_entropy = torch.sum(p * logp, dim=-1)
        return neg_entropy, p

    def calculate_lambda_tensor(self, neg_entropy, mask_positions):
        """Calculate lambda tensor from negative entropy"""
        if neg_entropy is None or self.scale is None:
            return None
        lambda_tensor = neg_entropy
        lambda_tensor = self.scale * torch.sigmoid(self.steepness * (lambda_tensor - self.centre))
        lambda_tensor = torch.where(mask_positions, lambda_tensor,
                                        torch.zeros_like(lambda_tensor))     
        return lambda_tensor.unsqueeze(-1)  # (B,T,1)



    def get_only_topk_probs(self, logits, mixinputs_k=3):
        """Mix only top-k embeddings based on their softmax probabilities (after topk selection)"""

        topk_logits, topk_indices = torch.topk(logits, k=mixinputs_k, dim=-1)  # (batch_size, seq_len, k)

        topk_probs = torch.softmax(topk_logits, dim=-1)  # (batch_size, seq_len, k)
        topk_sum = topk_probs.sum(dim=-1)  # (batch_size, seq_len)
        assert torch.allclose(topk_sum, torch.ones_like(topk_sum), atol=1e-1), \
            f"Top-k softmax probabilities do not sum to 1: max deviation = {(topk_sum - 1).abs().max().item()}"
    
        probs_full = torch.zeros_like(logits)                                 # (B, L, V)
        probs_full.scatter_(-1, topk_indices, topk_probs)                     # fill top-k
        assert torch.sum(probs_full > 0).item() == mixinputs_k * logits.shape[0] * logits.shape[1], \
            f"Number of non-zero entries in probs_full is incorrect: got {torch.sum(probs_full > 0).item()}, expected {mixinputs_k * logits.shape[0] * logits.shape[1]}"
        
        return probs_full
    

### ALL functions below are utility functions to attach/detach transparency head to/from a model ###

def attach_transparency(model, trans_args={}):
    mask_token_id = model.config.mask_token_id
    model.transparency = TransparencyHead(mask_token_id, trans_args)

def _base(model):
    """Get the base *HF* model that owns .transparency"""
    m = getattr(model, "get_base_model", lambda: model)()
    return m 

def transparency_head(model):
    """Get the transparency head from the model, if it exists"""
    return getattr(_base(model), "transparency", None)

def require_grad_for_th(model):
    """Require grad for transparency head parameters, except embed"""
    th = transparency_head(model)
    if th is not None:
        for p in th.parameters(recurse=False):
            p.requires_grad_(True)

def get_th_kwargs(model, verbose=False):
    """Get transparency head parameters as a dictionary to save in config"""
     # unwrap PEFT base (if present)
    th = transparency_head(model)
    if th is None:
        return {
            "transparency_alg": "none"
        }
    
    dtype  = torch.bfloat16
    th_params = {
        "transparency_alg": th.transparency_alg,
        "transparency_scale": th.scale.to(dtype).item(),
        "transparency_steepness": th.steepness.to(dtype).item(),
        "transparency_centre": th.centre.to(dtype).item(),
        "mixture_temp": th.temperature.to(dtype).item(),
        "mixinputs_k": th.mixinputs_k,
        "interpolation": th.interpolation,
        "transparency_scheduling": "none",  # we dont train with time_dependence
    }
    # if verbose:
    #     print(f"Transparency head params: {th_params}", flush=True)
    return th_params