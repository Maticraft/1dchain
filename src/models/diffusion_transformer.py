# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# https://github.com/facebookresearch/DiT/blob/main/LICENSE.txt
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------


import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp

from src.models.diffusion import DiffusionModel
from src.models.majorana_representation_encoder import MajoranaRepresentationPatchEmbed
from src.models.majorana_representation_generator import MajoranaRepresentationUnpatch


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class FreqEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256, period=10000):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.period = period

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size, self.period)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(DiffusionModel):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        input_channels=1,
        output_size=32,
        output_channels=1,
        patch_size=2,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        min_inter_site_interaction_range=2,
        max_inter_site_interaction_range=3,
        timesteps=500,
        input_embedder='patch',
        output_unpatcher='hamiltonian',
        **kwargs
    ):
        super().__init__()
        self.input_embedder = input_embedder
        self.output_unpatcher = output_unpatcher
        self.learn_sigma = learn_sigma
        # self.in_channels = in_channels
        # self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        if self.input_embedder == 'patch':
            self.x_embedder = PatchEmbed(input_size, self.patch_size, input_channels, hidden_size, bias=True)
            self.num_patches = self.x_embedder.num_patches
            self.seq_size = input_size // patch_size

        if self.input_embedder == 'hamiltonian':
            # works only for patch size = 4
            assert input_channels == 1, 'Hamiltonian embedder only works for 1 channel'
            self.hamiltonian_embeddder = MajoranaRepresentationPatchEmbed(hidden_size, min_inter_site_interaction_range, max_inter_site_interaction_range)
            interaction_range = max_inter_site_interaction_range - min_inter_site_interaction_range        
            self.seq_size = input_size // 4
            self.num_patches = self.seq_size * (interaction_range * self.hamiltonian_embeddder.num_inter_site_params + self.hamiltonian_embeddder.num_on_site_params)
            # total_params = self.hamiltonian_embeddder.num_on_site_params + (self.hamiltonian_embeddder.num_inter_site_params * interaction_range)

        self.t_embedder = FreqEmbedder(hidden_size, period=timesteps)
        # self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        self.polarization_embedder = FreqEmbedder(hidden_size // 2, period=0.5)
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])

        if self.output_unpatcher == 'patch':
            self.in_to_out_coverter = nn.Linear(self.num_patches, (output_size // self.patch_size)**2)
            self.final_layer = FinalLayer(hidden_size, int(mlp_ratio * (self.patch_size**2)))
            self.conv_unpatch = nn.ConvTranspose2d(int(mlp_ratio * (self.patch_size**2)), output_channels, kernel_size=patch_size, stride=patch_size)

        if self. output_unpatcher == 'hamiltonian':
            assert output_channels == 1, 'Hamiltonian unpatcher only works for 1 channel'
            self.hamiltonian_unpatch = MajoranaRepresentationUnpatch(mlp_ratio, min_inter_site_interaction_range, max_inter_site_interaction_range, **kwargs)
            self.in_to_out_coverter = nn.Linear(self.num_patches, (output_size // self.hamiltonian_unpatch.block_size) * self.hamiltonian_unpatch.num_total_params)
            self.final_layer = FinalLayer(hidden_size, int(mlp_ratio*self.hamiltonian_unpatch.num_total_params))
        
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.seq_size, self.num_patches // self.seq_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        if self.input_embedder == 'patch':
            w = self.x_embedder.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        if self.input_embedder == 'hamiltonian':
            w = self.hamiltonian_embeddder.on_site_converter.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            # nn.init.constant_(self.hamiltonian_embeddder.on_site_converter.bias, 0)

            for conv in self.hamiltonian_embeddder.inter_site_converters:
                w = conv.weight.data
                nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
                # nn.init.constant_(conv.bias, 0)

            # Initialize label embedding table:
            nn.init.normal_(self.hamiltonian_embeddder.on_site_embedding_table.weight, std=0.02)
            nn.init.normal_(self.hamiltonian_embeddder.inter_site_embedding_table.weight, std=0.02)

        # Initialize polarization embedding MLP:
        nn.init.normal_(self.polarization_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.polarization_embedder.mlp[2].weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        w = self.in_to_out_coverter.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        if self.output_unpatcher == 'patch':
            w = self.conv_unpatch.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
        if self.output_unpatcher == 'hamiltonian':
            # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
            w = self.hamiltonian_unpatch.on_site_converter.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            # nn.init.constant_(self.hamiltonian_unpatch.on_site_converter.bias, 0)

            for conv in self.hamiltonian_unpatch.inter_site_converters:
                w = conv.weight.data
                nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
                # nn.init.constant_(conv.bias, 0)

    def forward(self, x, time_step, context_vector):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        time_step: (N, 1) tensor of diffusion timesteps
        context_vector: (N, 2) tensor of polarization (left and right)
        """
        if self.input_embedder == 'patch':
            x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        if self.input_embedder == 'hamiltonian':
            x = self.hamiltonian_embeddder(x) + self.pos_embed
        t = self.t_embedder(time_step.squeeze(-1))                   # (N, D)
        # y = self.y_embedder(context_vector.long().squeeze(-1), self.training)    # (N, D)
        # y_0 = self.polarization_embedder(context_vector[..., 0])    # (N, D / 2)
        # y_1 = self.polarization_embedder(context_vector[..., 1])    # (N, D / 2)
        # y = torch.cat([y_0, y_1], dim=-1)                           # (N, D)
        c = t  #+ y                              # (N, D)
        for block in self.blocks:
            x = block(x, c)                      # (N, T, D)
        x = self.in_to_out_coverter(x.transpose(1, 2)).transpose(1, 2)  # (N, T2, D)
        x = self.final_layer(x, c)                # (N, T2, total_params)
        
        if self.output_unpatcher == 'patch':
            dim = int(x.shape[1] ** 0.5)
            x = torch.unflatten(x, dim=1, sizes=(dim, dim))  # (N, H2, W2, total_params)
            x = x.moveaxis(-1, 1)                           # (N, total_params, H2, W2)
            x = self.conv_unpatch(x)                # (N, out_channels, H, W)
        if self.output_unpatcher == 'hamiltonian':
            x = self.hamiltonian_unpatch(x)                 # (N, out_channels, H, W)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size_h, grid_size_w, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size_h, dtype=np.float32)
    grid_w = np.arange(grid_size_w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size_h, grid_size_w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}