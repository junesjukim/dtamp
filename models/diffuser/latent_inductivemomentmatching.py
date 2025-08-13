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
This code implements Inductive Moment Matching (IMM) in latent space,
adapting the structure of LatentDiffusion/LatentFlowMatching for full
compatibility (method names, signatures, and config usage).
'''

def apply_conditioning(x, cond):
    for t, val in cond.items():
        x[:, t, :] = val.clone()
    return x


class LatentInductiveMomentMatching(nn.Module):
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
        print(f"IMM model configured to predict: {self.prediction_type}")
        self.theta_min = 1e-3
        # EMA target for IMM target velocity/path
        self.ema_tau = 1e-3
        self.model_target = self._build_ema_model(self.model)

        ## get loss coefficients and initialize objective
        if loss_type == 'l1':
            self.loss_fn = lambda x, y: (x - y).abs()
        elif loss_type == 'l2':
            self.loss_fn = lambda x, y: (x - y).pow(2)

    def _build_ema_model(self, model: nn.Module) -> nn.Module:
        import copy
        ema = copy.deepcopy(model)
        for p in ema.parameters():
            p.requires_grad_(False)
        return ema

    @torch.no_grad()
    def _polyak_update(self) -> None:
        tau = self.ema_tau
        for p, tp in zip(self.model.parameters(), self.model_target.parameters()):
            tp.data.mul_(1.0 - tau).add_(tau * p.data)

    #------------------------------------------ sampling ------------------------------------------#

    def p_mean_variance(self, x, cond, t, returns=None):
        # Deterministic Euler step for IMM
        assert self.n_timesteps % self.n_sample_timesteps == 0, f"n_timesteps({self.n_timesteps}) must be divisible by n_sample_timesteps({self.n_sample_timesteps})"
        t_model = t * (self.n_timesteps // self.n_sample_timesteps)

        if self.returns_condition:
            out_cond = self.model(x, cond, t_model, returns, use_dropout=False)
            out_uncond = self.model(x, cond, t_model, returns, force_dropout=True)
            model_output = out_uncond + self.condition_guidance_w * (out_cond - out_uncond)
        else:
            model_output = self.model(x, cond, t_model, returns)

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

        return x_less_noisy

    @torch.no_grad()
    def p_sample(self, x, cond, t, returns=None):
        # In flow matching, the sampling step is deterministic given the model's output.
        # We directly use the calculated mean as the next state.
        model_mean = self.p_mean_variance(x=x, cond=cond, t=t, returns=returns)
        
        # Apply conditioning to the updated state
        x_out = apply_conditioning(model_mean, cond)
        return x_out

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

    def _kernel_rbf(self, x, y, w):
        # x,y: [B, M, D], w: [B,1,1]
        diff = x.unsqueeze(2) - y.unsqueeze(1)  # [B, M, M, D]
        norm = torch.norm(diff, dim=-1).clamp(min=1e-4)  # [B, M, M]
        exponent = - w * norm / x.size(-1)
        return torch.exp(exponent)

    def _loss_mmd(self, x, y, w):
        k_xx = self._kernel_rbf(x, x, w)
        k_yy = self._kernel_rbf(y, y, w)
        k_xy = self._kernel_rbf(x, y, w)
        return (k_xx.mean(dim=[1,2]) + k_yy.mean(dim=[1,2]) - 2.0 * k_xy.mean(dim=[1,2])).mean()

    def p_losses(self, x_start, cond, t, returns=None):
        # IMM loss: MMD between x_{s|t} and x_{s|r} with EMA target
        device = x_start.device
        B, H, D = x_start.shape

        # t ~ U[0,1), s=r=1 (타깃은 최종 상태)
        t_cont = torch.rand(B, 1, 1, device=device)
        s_cont = torch.ones_like(t_cont)
        r_cont = s_cont.clone()

        # 연속시간 → 정수 스텝 변환
        t_model = (t_cont.squeeze(-1).squeeze(-1) * self.n_timesteps).clamp(0, self.n_timesteps - 1).long()
        r_model = (r_cont.squeeze(-1).squeeze(-1) * self.n_timesteps).clamp(0, self.n_timesteps - 1).long()

        # 경로 샘플링 (공유 noise)
        noise = torch.randn_like(x_start)
        x_t = (1.0 - (1.0 - self.theta_min) * t_cont) * noise + t_cont * x_start
        x_r = (1.0 - (1.0 - self.theta_min) * r_cont) * noise + r_cont * x_start

        # 모델 출력 → 속도장 v(x,t)
        if self.returns_condition:
            out_c = self.model(x_t, cond, t_model, returns, use_dropout=False)
            out_u = self.model(x_t, cond, t_model, returns, force_dropout=True)
            model_out_t = out_u + self.condition_guidance_w * (out_c - out_u)
        else:
            model_out_t = self.model(x_t, cond, t_model, returns)

        with torch.no_grad():
            model_out_r = self.model_target(x_r, cond, r_model, returns)

        # x_start 모드면 x₁_hat → v(x,t) 변환
        if self.prediction_type == "x_start":
            denom_t = (1.0 - (1.0 - self.theta_min) * t_cont).clamp(min=1e-8)
            denom_r = (1.0 - (1.0 - self.theta_min) * r_cont).clamp(min=1e-8)
            v_t = (model_out_t - (1.0 - self.theta_min) * x_t) / denom_t
            v_r = (model_out_r - (1.0 - self.theta_min) * x_r) / denom_r
        else:
            v_t = model_out_t
            v_r = model_out_r

        # IMM 업데이트: x_{s|t} = x_t + v(x_t,t)(s-t)
        delta_st = (s_cont - t_cont).clamp(min=1e-4)
        delta_sr = (s_cont - r_cont).clamp(min=1e-4)
        x_st = x_t + v_t * delta_st
        x_sr = x_r + v_r * delta_sr

        # 조건 타임스텝 마스킹
        if len(cond) > 0:
            all_idx = torch.arange(H, device=device)
            cond_idx_list = []
            for k in cond.keys():
                cond_idx_list.append(k if k >= 0 else H + k)
            if len(cond_idx_list) > 0:
                cond_idx = torch.tensor(cond_idx_list, device=device, dtype=torch.long)
                mask_idx = all_idx[~torch.isin(all_idx, cond_idx)]
            else:
                mask_idx = all_idx
            x_st = x_st.index_select(dim=1, index=mask_idx)
            x_sr = x_sr.index_select(dim=1, index=mask_idx)

        # MMD 손실: k(x,y) = exp(-w||x-y||/D), w = 1/(s-t)
        w = 1.0 / delta_st
        loss = self._loss_mmd(x_st, x_sr, w)

        with torch.no_grad():
            self._polyak_update()

        return loss

    def loss(self, x, cond, returns=None):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, cond, t, returns)

    def forward(self, cond, returns, *args, **kwargs):
        return self.conditional_sample(cond=cond, returns=returns, *args, **kwargs)
