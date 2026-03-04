#
# Copyright 2026- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

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
    def __init__(self, mask_token_id, trans_args):
        super().__init__()
            
        self.mask_token_id = mask_token_id

        # Initial scales
        init_scale = getattr(trans_args, "init_scale", 0.0)
        init_centre = getattr(trans_args, "init_centre", -0.75)
        init_steep = getattr(trans_args, "init_steep", 10/1.5)
        init_temperature = getattr(trans_args, "init_temperature", 1.0)

        # Scales with transformations to stay in boundaries
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
        
        # apply only on mask positions
        lambda_tensor = torch.where(mask_positions, lambda_tensor,
                                        torch.zeros_like(lambda_tensor))     
        return lambda_tensor

    
    def forward(self, input_ids, logits_prelim, embed_weight=None):
        """
        Args:
            input_ids:      (B, T) token ids.
            logits_prelim:  (B, T, V) logits from the preliminary pass.
            embed_weight:   (V, D) embedding weight matrix. Required when
                            interpolation == "spherical"; ignored otherwise.
        Returns:
            If interpolation == "linear":  (final_indices, final_probs) or (B,T,V).
            If interpolation == "spherical": (B, T, D) dense embeddings.
        """
        
        # --- 1. Get Entropy and Lambda (No change) ---
        temperature = self.temperature if self.transparency_alg == "mixinputs_with_temp" else 1.0
        neg_entropy, p_full = self.get_neg_entropy_and_probabilities(logits_prelim, temperature=temperature)

        mask_positions = (input_ids == self.mask_token_id)
        
        lambda_tensor = self.calculate_lambda_tensor(neg_entropy, mask_positions)
        lambda_tensor = lambda_tensor.unsqueeze(-1)  # (B, T, 1)
        lambda_tensor[~mask_positions] = 0.0

        # --- 2. Branch on interpolation mode ---
        if self.interpolation == "spherical":
            assert embed_weight is not None, \
                "embed_weight must be provided for spherical interpolation"
            # For SLERP we need a full (B,T,V) probability distribution
            if self.transparency_alg == "mixinputs_with_topk":
                p_for_slerp = self._get_topk_full_probs(logits_prelim)
            else:
                p_for_slerp = p_full
            return self._slerp_interpolate(input_ids, p_for_slerp, lambda_tensor, embed_weight)

        # --- Original linear paths below ---
        if self.transparency_alg == "mixinputs_with_topk":
            # GATHER: Select only the logits for masked positions
            masked_logits = logits_prelim[mask_positions]
            
            if masked_logits.shape[0] > 0:
                # COMPUTE: Get top-k indices and probs for masked items
                topk_indices_masked, topk_probs_masked = self.get_only_topk_probs(
                    masked_logits, self.mixinputs_k
                )
                
                # SCATTER: Create full (B, T, k) tensors
                topk_indices = torch.zeros(
                    (input_ids.shape[0], input_ids.shape[1], self.mixinputs_k), 
                    dtype=topk_indices_masked.dtype, device=input_ids.device
                )
                topk_probs = torch.zeros_like(topk_indices, dtype=topk_probs_masked.dtype)
                
                topk_indices[mask_positions] = topk_indices_masked
                topk_probs[mask_positions] = topk_probs_masked
                
            else:
                # No masks, just create empty tensors
                topk_indices = torch.zeros(
                    (input_ids.shape[0], input_ids.shape[1], self.mixinputs_k), 
                    dtype=torch.long, device=input_ids.device
                )
                topk_probs = torch.zeros_like(topk_indices, dtype=logits_prelim.dtype)

            # Component 1: The original one-hot token
            indices_onehot = input_ids.unsqueeze(-1) 
            probs_onehot = (1.0 - lambda_tensor) 

            # Component 2: The top-k predictions
            indices_topk = topk_indices 
            probs_topk = lambda_tensor * topk_probs # Scale by lambda

            # Concatenate into final (B, T, k+1) tensors
            final_indices = torch.cat([indices_onehot, indices_topk], dim=-1)
            final_probs = torch.cat([probs_onehot, probs_topk], dim=-1)

            # Return the two sparse tensors as a tuple
            return (final_indices, final_probs)
            
        else:
            xt_one_hot = F.one_hot(input_ids, num_classes=logits_prelim.shape[-1]).to(logits_prelim.dtype)
            
            # p_out shape is (B, T, V)
            p_out = (1 - lambda_tensor) * xt_one_hot \
                    + lambda_tensor * p_full

            return p_out

    def _get_topk_full_probs(self, logits):
        """
        Build a full (B, T, V) sparse probability distribution from top-k.
        Used by the spherical path to compute e_pred = p @ W.
        """
        topk_logits, topk_indices = torch.topk(
            logits, k=self.mixinputs_k, dim=-1)  # (B, T, k)
        topk_logits = topk_logits.to(torch.float32)
        topk_probs = torch.softmax(topk_logits, dim=-1).to(logits.dtype)  # (B, T, k)
        probs_full = torch.zeros_like(logits)  # (B, T, V)
        probs_full.scatter_(-1, topk_indices, topk_probs)
        return probs_full

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


    def get_only_topk_probs(self, logits, mixinputs_k=3):
        """
        NEW: Returns (topk_indices, topk_probs)
        Shape: (B, T, k), (B, T, k)
        """
        topk_logits, topk_indices = torch.topk(logits, k=mixinputs_k, dim=-1)
        topk_logits = topk_logits.to(torch.float32)
        
        topk_probs = torch.softmax(topk_logits, dim=-1)
        
        # Return the components, not the full tensor
        return topk_indices, topk_probs.to(logits.dtype)

