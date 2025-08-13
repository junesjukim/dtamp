import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
from torch.distributions import Bernoulli

from .helpers import (
    SinusoidalPosEmb,
    Downsample1d,
    Upsample1d,
    Conv1dBlock,
)

# Reuse residual temporal block implementation from temporal.py to avoid duplication
from .temporal import ResidualTemporalBlock


class TemporalUnetTimeAdd(nn.Module):
    """
    Temporal U-Net variant that conditions on two time inputs: t and h = t - r.

    This module mirrors the architecture of `TemporalUnet` in `temporal.py`, but
    extends the time embedding pathway to separately embed `t` and `h`, then
    concatenates the two embeddings (and optional returns embedding) to form the
    conditioning vector supplied to all residual temporal blocks.

    Forward signature uses floating time inputs (normalized to [0, 1]):
        - time_t: tensor of shape [batch]
        - time_h: tensor of shape [batch]
    """

    def __init__(
        self,
        horizon,
        transition_dim,
        dim: int = 128,
        dim_mults=(1, 2, 4, 8),
        returns_condition: bool = False,
        condition_dropout: float = 0.1,
        calc_energy: bool = False,
        kernel_size: int = 5,
    ):
        super().__init__()

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f'[ models/temporal ] Channel dimensions: {in_out}')

        # Activation selection to match original TemporalUnet behavior
        if calc_energy:
            mish = False
            act_fn = nn.SiLU()
        else:
            mish = True
            act_fn = nn.Mish()

        self.time_dim = dim
        self.returns_dim = dim
        # Shared embedding pipeline for both t and h (weight sharing)
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            act_fn,
            nn.Linear(dim * 4, dim),
        )

        self.returns_condition = returns_condition
        self.condition_dropout = condition_dropout
        self.calc_energy = calc_energy
        if self.returns_condition:
            self.returns_mlp = nn.Sequential(
                        nn.Linear(1, dim),
                        act_fn,
                        nn.Linear(dim, dim * 4),
                        act_fn,
                        nn.Linear(dim * 4, dim),
                    )
            self.mask_dist = Bernoulli(probs=1-self.condition_dropout)
            embed_dim = dim + dim + dim  # t_emb + h_emb + returns_emb
        else:
            embed_dim = dim + dim  # t_emb + h_emb

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)
        print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size, mish=mish),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size, mish=mish),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon // 2

        # Bottleneck
        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size, mish=mish)
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size, mish=mish)

        
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size, mish=mish),
                ResidualTemporalBlock(dim_in, dim_in, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size, mish=mish),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon * 2

        # Final projection back to transition_dim
        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=kernel_size, mish=mish),
            nn.Conv1d(dim, transition_dim, 1),
        )



    def forward(self, x, cond, time_t, time_h, returns=None, use_dropout=True, force_dropout=False):
        """
        x:       [batch, horizon, transition]
        returns: [batch, horizon] if returns_condition else None
        time_t:  [batch]
        time_h:  [batch]
        """
        if self.calc_energy:
            x_inp = x

        x = einops.rearrange(x, 'b h t -> b t h')
        
        t_emb = self.time_mlp(time_t)
        h_emb = self.time_mlp(time_h)
        emb = torch.cat([t_emb, h_emb], dim=-1)

        if self.returns_condition:
            assert returns is not None
            returns_embed = self.returns_mlp(returns)
            if use_dropout:
                mask = self.mask_dist.sample(sample_shape=(returns_embed.size(0), 1)).to(returns_embed.device)
                returns_embed = mask*returns_embed
            if force_dropout:
                returns_embed = 0*returns_embed
            emb = torch.cat([emb, returns_embed], dim=-1)
        
        h = []
        for resnet, resnet2, downsample in self.downs:
            x = resnet(x, emb)
            x = resnet2(x, emb)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, emb)
        x = self.mid_block2(x, emb)

        for resnet, resnet2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, emb)
            x = resnet2(x, emb)
            x = upsample(x)

        x = self.final_conv(x)
        x = einops.rearrange(x, 'b t h -> b h t')

        if self.calc_energy:
            energy = ((x - x_inp)**2).mean()
            grad = torch.autograd.grad(outputs=energy, inputs=x_inp, create_graph=True)
            return grad[0]
        else:
            return x

    def get_pred(self, x, cond, time_t, time_h, returns=None, use_dropout=True, force_dropout=False):
        """
        Prediction path without energy gradients. Signature mirrors `forward`.
        """
        x = einops.rearrange(x, 'b h t -> b t h')

        t_emb = self.time_mlp(time_t)
        h_emb = self.time_mlp(time_h)
        emb = torch.cat([t_emb, h_emb], dim=-1)

        if self.returns_condition:
            assert returns is not None
            returns_embed = self.returns_mlp(returns)
            if use_dropout:
                mask = self.mask_dist.sample(sample_shape=(returns_embed.size(0), 1)).to(returns_embed.device)
                returns_embed = mask*returns_embed
            if force_dropout:
                returns_embed = 0*returns_embed
            emb = torch.cat([emb, returns_embed], dim=-1)
        
        h = []
        for resnet, resnet2, downsample in self.downs:
            x = resnet(x, emb)
            x = resnet2(x, emb)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, emb)
        x = self.mid_block2(x, emb)

        for resnet, resnet2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, emb)
            x = resnet2(x, emb)
            x = upsample(x)

        x = self.final_conv(x)
        x = einops.rearrange(x, 'b t h -> b h t')
        return x


