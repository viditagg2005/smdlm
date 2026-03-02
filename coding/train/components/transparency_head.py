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

    def forward(self, input_ids, logits_prelim):

         # Create a one-hot distribution for the original input `xt`.
        xt_one_hot = F.one_hot(input_ids, num_classes=logits_prelim.shape[-1]).to(logits_prelim.dtype)
        
        # First get the negative entropy to calculate lambda
        temperature = self.temperature if self.transparency_alg == "mixinputs_with_temp" else 1.0
        neg_entropy, p = self.get_neg_entropy_and_probabilities(logits_prelim, temperature=temperature)

        # Get mask positions
        mask_positions = (input_ids == self.mask_token_id)

        # calculate lambda tensor
        lambda_tensor = self.calculate_lambda_tensor(neg_entropy, mask_positions) # (B,T,1)

        if self.transparency_alg == "mixinputs_with_topk":
            # Mix only top-k embeddings based on their softmax probabilities (after topk selection)
            p = self.get_only_topk_probs(logits_prelim, self.mixinputs_k)

        # Create convex combination 
        p_out = (1 - lambda_tensor) * xt_one_hot \
                    + lambda_tensor * p

        return p_out

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
        "transparency_scheduling": "none",  # we dont train with time_dependence
    }
    # if verbose:
    #     print(f"Transparency head params: {th_params}", flush=True)
    return th_params