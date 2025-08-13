import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


from .helpers import (
    extract,
)

'''
This code implements MeanFlow in the latent trajectory setting while preserving the
public API used by LatentDiffusion/LatentFlowMatching. The design goal is minimum
external change:

- Same callable methods: loss, forward, conditional_sample, p_sample_loop, etc.
- Same constructor signature and flags to remain drop-in compatible.

Key differences from LatentFlowMatching:
- The model is trained to predict average velocity u(z_t, r=0, t).
- Training target uses the MeanFlow identity with JVP along z and a finite
  difference approximation for the explicit t-derivative term.
- Sampling integrates one-step (or few-step) updates: x <- x - Δt * u(x, t).
'''


def apply_conditioning(x, cond):
    for t, val in cond.items():
        x[:, t, :] = val.clone()
    return x


class LatentMeanFlow(nn.Module):
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

        # Flow bridge parameter used in q_sample (matches LatentFlowMatching)
        self.theta_min = 1e-3

        if loss_type == 'l1':
            self.loss_fn = lambda x, y: (x - y).abs()
        elif loss_type == 'l2':
            self.loss_fn = lambda x, y: (x - y).pow(2)

    #------------------------------------------ sampling ------------------------------------------#

    @torch.no_grad()
    def p_sample(self, x, cond, t_float, r_float, returns=None):
        # The model predicts average velocity u(z_t, r, t) with continuous times.
        # During sampling we assume CFG has been distilled at training time,
        # so we only need a single evaluation.
        batch_size = x.shape[0]
        device = x.device
        u = self.model(x, cond, t_float, r_float, returns)

        # MeanFlow update rule: z_r = z_t - (t-r) u(z_t, r, t)
        time_step_size = t_float - r_float
        x_next = x - time_step_size * u

        if self.clip_denoised:
            x_next.clamp_(-1., 1.)

        x_next = apply_conditioning(x_next, cond)
        return x_next

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, returns=None, return_diffusion=False):
        device = next(self.model.parameters()).device

        batch_size = shape[0]
        # Start from pure noise N(0, I)
        x = torch.randn(shape, device=device)
        x = apply_conditioning(x, cond)

        if return_diffusion: diffusion = [x]

        # Flowmatching samples forward in time from 0 to T-1
        t_float = torch.full((batch_size,), 1.0, device=device, dtype=torch.float)
        r_float = torch.full((batch_size,), 0.0, device=device, dtype=torch.float)
        x = self.p_sample(x, cond, t_float, r_float, returns)
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

    def sample_t_r(self, batch_size: int, device: torch.device):
        """
        Sample continuous t, r in [0,1] with t >= r. A portion has r == t,
        which recovers standard Flow Matching on those samples.
        """
        t_rand = torch.rand(batch_size, device=device)
        r_rand = torch.rand(batch_size, device=device)
        t_cont = torch.maximum(t_rand, r_rand)
        r_cont = torch.minimum(t_rand, r_rand)

        # enforce r == t for a fraction of the batch
        r_equal_t_ratio = 0.25
        num_equal = int(r_equal_t_ratio * batch_size)
        if num_equal > 0:
            idx = torch.randperm(batch_size, device=device)[:num_equal]
            r_cont[idx] = t_cont[idx]

        return t_cont, r_cont

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        # t is already continuous in [0,1]; reshape for broadcasting
        t_reshaped = t.view(-1, 1, 1).to(x_start.device)

        # Conditional Flow Matching: z_t = (1 - (1 - theta_min) * t) * z0 + t * z1
        sample = (1 - (1 - self.theta_min) * t_reshaped) * noise + t_reshaped * x_start
        return sample


    def p_losses(self, x_start, cond, t_cont, r_cont, returns=None):
        noise = torch.randn_like(x_start)

        # x_noisy from continuous time directly
        x_noisy = self.q_sample(x_start=x_start, t=t_cont, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond)

        # derivative of z(t) wrt normalized time
        v_t = x_start - (1 - self.theta_min) * noise

        # continuous times and interval
        h_cont = t_cont - r_cont

        # Define fn(z, r, t) for a single JVP over (v, 0, 1)
        def fn_all(z, r, t):
            return self.model(z, cond, t, t - r, returns)

        # Single JVP to get (u, dudt = v·∂_z u + ∂_t u)
        u_pred, dudt = torch.autograd.functional.jvp(
            fn_all,
            (x_noisy, r_cont, t_cont),
            (v_t.detach(), torch.zeros_like(r_cont), torch.ones_like(t_cont)),
            create_graph=True,
        )

        # 6) MeanFlow target with general (t, r)
        delta_tr = (t_cont - r_cont).view(-1, 1, 1)
        u_tgt = v_t - delta_tr * dudt

        # 7) loss with masking on conditioned positions
        u_for_loss = u_pred.clone()
        u_tgt_for_loss = u_tgt.detach().clone()
        for k in cond.keys():
            u_for_loss[:, k, :] = 0
            u_tgt_for_loss[:, k, :] = 0

        loss = self.loss_fn(u_for_loss, u_tgt_for_loss).mean()
        return loss

    def loss(self, x, cond, returns=None):
        batch_size = len(x)
        device = x.device
        t, r = self.sample_t_r(batch_size, device)
        
        return self.p_losses(x, cond, t, r, returns)

    def forward(self, cond, returns, *args, **kwargs):
        return self.conditional_sample(cond=cond, returns=returns, *args, **kwargs)


