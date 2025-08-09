import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import pdb
import math

from .helpers import (
    extract,
)

'''
This code is adapted for flow matching based on the structure of LatentDiffusion.
The core flow matching logic is derived from the user-provided reference code.
'''

def apply_conditioning(x, cond):
    for t, val in cond.items():
        x[:, t, :] = val.clone()
    return x


class LatentFlowMatching(nn.Module):
    def __init__(self, model, horizon, latent_dim, n_timesteps=1000, n_sample_timesteps=1,
        loss_type='l2', clip_denoised=False, predict_epsilon=True, returns_condition=False, condition_guidance_w=0.1):
        super().__init__()
        self.horizon = horizon
        self.transition_dim = latent_dim
        self.model = model
        self.returns_condition = returns_condition
        self.condition_guidance_w = condition_guidance_w
        self.n_timesteps = int(n_timesteps)
        self.n_sample_timesteps = int(n_sample_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        # For flowmatching, we predict 'x_start' or 'velocity'
        self.prediction_type = "x_start" if self.predict_epsilon else "velocity"
        print(f"Flowmatching model configured to predict: {self.prediction_type}")
        self.theta_min = 1e-3

        ## get loss coefficients and initialize objective
        if loss_type == 'l1':
            self.loss_fn = lambda x, y: (x - y).abs()
        elif loss_type == 'l2':
            self.loss_fn = lambda x, y: (x - y).pow(2)

    #------------------------------------------ sampling ------------------------------------------#

    def p_mean_variance(self, x, cond, t, returns=None):
        # The naming is kept for consistency, but for flow matching, we only compute the mean (the next state).
        # NOTE: The time input to the model is scaled similarly to the original LatentDiffusion for consistency.
        
        model_output = self.model(x, cond, t * (self.n_timesteps // self.n_sample_timesteps), returns)
        model_output = apply_conditioning(model_output, cond)
        
        if self.prediction_type == "x_start":
            t_normalized = t.float() / self.n_sample_timesteps
            t_normalized = t_normalized.view(-1, 1, 1).to(x.device)
            # Perform the update step of the Euler integrator
            x_less_noisy = x + (model_output - (1.0-self.theta_min) * x) / (1.0 - (1.0-self.theta_min) * t_normalized + 1e-8) * (1.0 / self.n_sample_timesteps)
        else:  # velocity prediction
            # Perform the update step of the Euler integrator
            x_less_noisy = x + model_output * (1.0 / self.n_sample_timesteps)
        
        # This clamping is not part of the standard flow matching but kept for stability.
        if self.clip_denoised:
            x_less_noisy.clamp_(-1., 1.)

        x_less_noisy = apply_conditioning(x_less_noisy, cond)
        return x_less_noisy

    @torch.no_grad()
    def p_sample(self, x, cond, t, returns=None):
        # In flow matching, the sampling step is deterministic given the model's output.
        # We directly use the calculated mean as the next state.
        model_mean = self.p_mean_variance(x=x, cond=cond, t=t, returns=returns)
        return model_mean

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, returns=None, return_diffusion=False):
        device = next(self.model.parameters()).device

        batch_size = shape[0]
        # Start from pure noise N(0, I)
        x = torch.randn(shape, device=device)
        x = apply_conditioning(x, cond)

        if return_diffusion: diffusion = [x]

        # Flowmatching samples forward in time from 0 to T-1
        for i in range(0, self.n_sample_timesteps):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, cond, timesteps, returns)
            x = apply_conditioning(x, cond)

            if return_diffusion: diffusion.append(x)

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    @torch.no_grad()
    def conditional_sample(self, cond, returns=None, horizon=None, *args, **kwargs):
        device = next(self.model.parameters()).device
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)

        return self.p_sample_loop(shape, cond, returns, *args, **kwargs)

    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Correctly handle device placement for t_normalized
        t_normalized = (t.float() / self.n_timesteps).view(-1, 1, 1).to(x_start.device)

        # Conditional Flow Matching: x_t = (1 - (1 - self.theta_min) * t) * x_0 + t * x_1
        # Here, x_0 is noise and x_1 is the data (x_start)
        sample = (1 - (1 - self.theta_min) * t_normalized) * noise + t_normalized * x_start
        return sample

    def p_losses(self, x_start, cond, t, returns=None):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond)

        model_output = self.model(x_noisy, cond, t, returns)

        
        if self.prediction_type == "x_start":
            target = x_start
        else: # velocity
            # The true velocity: v(x_t, t) = x_1 - (1 - theta_min)*x_0
            v_t_true = x_start - (1 - self.theta_min) * noise
            target = v_t_true
        
        # As in the original LatentDiffusion, we don't calculate loss on conditioned parts.
        # This is done by setting the target and prediction to zero at those points.
        if self.prediction_type == "x_start":
            model_output = apply_conditioning(model_output, cond)
            target = apply_conditioning(target, cond)
        else:
             for k in cond.keys():
                model_output[:, k, :] = 0
                target[:,k,:] = 0

        loss = self.loss_fn(model_output, target)
        return loss.mean()

    def loss(self, x, cond, returns=None):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, cond, t, returns)

    def forward(self, cond, returns, *args, **kwargs):
        return self.conditional_sample(cond=cond, returns=returns, *args, **kwargs)
