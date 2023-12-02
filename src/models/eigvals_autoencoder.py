import typing as t

import torch
import torch.nn as nn

from src.models.positional_autoencoder import PositionalDecoder, PositionalEncoder


class EigvalsPositionalEncoder(nn.Module):
    def __init__(self, input_size: t.Tuple[int, int, int], representation_dim: t.Union[int, t.Tuple[int, int]], **kwargs: t.Dict[str, t.Any]) -> None:
        super(EigvalsPositionalEncoder, self).__init__()
        self.channel_num = input_size[0]
        self.N = input_size[1]
        self.block_size = input_size[2]
        self.eigvals_num = kwargs.get('eigvals_num', self.N * self.block_size)
        self.eigvals_mlp_layers = kwargs.get('eigvals_mlp_layers', 3)
        self.pos_encoder = PositionalEncoder(input_size, representation_dim, **kwargs)
        self.eigvals_encoder = nn.Sequential(
            PositionalEncoder(input_size, self.eigvals_num, **kwargs),
            self._get_mlp(self.eigvals_mlp_layers, self.eigvals_num, self.eigvals_num, self.eigvals_num)
        )

    def _get_mlp(self, layers_num: int, input_size: int, hidden_size: int, output_size: int):
        layers = []
        for i in range(layers_num):
            if i == 0:
                layers.append(nn.Linear(input_size, hidden_size))
            elif i == layers_num - 1:
                layers.append(nn.BatchNorm1d(hidden_size))
                layers.append(nn.Linear(hidden_size, output_size))
            else:
                layers.append(nn.BatchNorm1d(hidden_size))
                layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.LeakyReLU(negative_slope=0.01))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        z = self.pos_encoder(x)
        eigvals = self.eigvals_encoder(x)
        return z, eigvals


class EigvalsPositionalDecoder(PositionalDecoder):
    def __init__(self, representation_dim: t.Union[int, t.Tuple[int, int]], output_size: t.Tuple[int, int, int], **kwargs: t.Dict[str, t.Any]):
        super(EigvalsPositionalDecoder, self).__init__(representation_dim, output_size, **kwargs)

        self.eigvals_num = self.N * self.block_size

        self.eigvals_encoder = nn.Sequential(
            self._get_mlp(4, self.eigvals_num, self.eigvals_num, self.eigvals_num),
            nn.Unflatten(1, (1, self.eigvals_num)),
            nn.Conv1d(1, self.kernel_num, kernel_size=self.block_size, stride=self.block_size),
            self._get_activation()
        )

        self.block_eigvals_combiner = nn.Sequential(
            nn.Conv1d(2*self.kernel_num, self.kernel_num, kernel_size=1),
            self._get_activation(),
        )

        self.naive_eigvals_seq_encoder = self._get_mlp(4, self.eigvals_num, self.eigvals_num, self.strip_len)

        self.tf_seq_decoder_layer = nn.TransformerEncoderLayer(d_model=self.kernel_num, nhead=2, dim_feedforward=128, batch_first=True)
        self.tf_seq_decoder = nn.TransformerEncoder(self.tf_seq_decoder_layer, num_layers=1)

    def _get_conv_block(self):
        return nn.Sequential(
            nn.ConvTranspose2d(2*self.kernel_num+2, self.channel_num, kernel_size=self.kernel_size, stride=self.stride, dilation=self.dilation),
        )

    def forward(self, latent_tuple: t.Tuple[torch.Tensor, torch.Tensor]):
        x, eigvals = latent_tuple
        block = self.naive_block_decoder(x[:, self.freq_dim:]).unsqueeze(0)
        block_expand = block.expand(self.strip_len, -1, -1).permute((1, 2, 0)) # shape (batch_size, kernel_num, strip_len)

        eigvals_enc = self.eigvals_encoder(eigvals) # shape (batch_size, kernel_num, strip_len)

        block_eigvals = self.block_eigvals_combiner(torch.cat([block_expand, eigvals_enc], dim=1)) # shape (batch_size, kernel_num, strip_len)

        freq = self.freq_decoder(x[:, :self.freq_dim])
        freq_seq = torch.stack([self._periodic_func(self.freq_seq_constructor[i](freq), i) for i in range(self.kernel_num)], dim=-1) # shape (batch_size, strip_len, kernel_num)

        tf_input = freq_seq * block_eigvals.transpose(1, 2) # shape (batch_size, strip_len, kernel_num)

        seq = self.tf_seq_decoder(tf_input) # shape (batch_size, strip_len, kernel_num)
        seq = seq.transpose(1, 2).unsqueeze(2) # shape (batch_size, kernel_num, 1, strip_len)

        naive_block_seq = self.naive_seq_decoder(x).unsqueeze(1).unsqueeze(1) # shape (batch_size, 1, 1, strip_len)
        naive_eigvals_seq = self.naive_eigvals_seq_encoder(eigvals).unsqueeze(1).unsqueeze(1) # shape (batch_size, 1, 1, strip_len)

        block_eigvals = block_eigvals.unsqueeze(2)  # shape (batch_size, kernel_num, 1, strip_len)
        block_seq = torch.cat([block_eigvals, seq, naive_block_seq, naive_eigvals_seq], dim=1) # shape (batch_size, 2*kernel_num+2, 1, strip_len)

        strips = self.conv(block_seq) # shape (batch_size, channel_num, block_size, block_size*N)
        matrix = self._get_matrix_from_strips(strips) # shape (batch_size, 2, block_size*N, block_size*N)
        return matrix