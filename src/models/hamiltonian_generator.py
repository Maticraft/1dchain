import typing as t
from itertools import product

import torch
import torch.nn as nn


class HamiltonianGenerator(nn.Module):
    def __init__(self, representation_dim: t.Union[int, t.Tuple[int, int]], output_size: t.Tuple[int, int, int], **kwargs: t.Dict[str, t.Any]):
        super(HamiltonianGenerator, self).__init__()
        self.channel_num = output_size[0]
        assert self.channel_num % 2 == 0, 'Channel number must be even'
        self.N = output_size[1]
        self.block_size = output_size[2]

        if type(representation_dim) == int:
            self.freq_dim = representation_dim // 2
            self.block_dim = representation_dim // 2
        else:
            self.freq_dim = representation_dim[0]
            self.block_dim = representation_dim[1]

        self.freq_dec_depth = kwargs.get('freq_dec_depth', 4)
        self.freq_dec_hidden_size = kwargs.get('freq_dec_hidden_size', 128)

        self.block_dec_depth = kwargs.get('block_dec_depth', 4)
        self.block_dec_hidden_size = kwargs.get('block_dec_hidden_size', 128)

        self.seq_dec_depth = kwargs.get('seq_dec_depth', 4)
        self.seq_dec_hidden_size = kwargs.get('seq_dec_hidden_size', 128)

        self.activation = kwargs.get('activation', 'relu')

        self.blocks, self.block_pairs = self._initialize_block_pairs()
        self.block_pair_idx_map = self._initialize_block_pair_idx_map()
        self.reduced_block_pairs = self.block_pairs
        if kwargs.get('reduce_blocks', False):
            self.reduced_block_pairs = [
                self.block_pairs[self.block_pair_idx_map['yy']],
                self.block_pairs[self.block_pair_idx_map['z1']],
                self.block_pairs[self.block_pair_idx_map['zx']],
                self.block_pairs[self.block_pair_idx_map['zy']],
            ]

        self.seq_num = 2*len(self.reduced_block_pairs) + self.channel_num - 2

        self.freq_decoder = self._get_mlp(self.freq_dec_depth, self.freq_dim, self.freq_dec_hidden_size, self.freq_dec_hidden_size)
        self.freq_seq_constructor = nn.ModuleList([
            self._get_mlp(self.freq_dec_depth, self.freq_dec_hidden_size, self.freq_dec_hidden_size, self.N, final_activation='sigmoid')
            for _ in range(self.seq_num)
        ])

        self.naive_block_decoder = self._get_mlp(self.block_dec_depth, self.block_dim, self.block_dec_hidden_size, self.seq_num)

        self.naive_seq_decoder = self._get_mlp(self.seq_dec_depth, self.freq_dim+self.block_dim, self.seq_dec_hidden_size, self.N)

        self.tf_seq_decoder_layer = nn.TransformerEncoderLayer(d_model=self.seq_num, nhead=2, dim_feedforward=128, batch_first=True)
        self.tf_seq_decoder = nn.TransformerEncoder(self.tf_seq_decoder_layer, num_layers=1)

        self.lin_mixer = nn.Conv2d(2*self.seq_num + 1, self.seq_num, kernel_size=1, stride=1)


    def _initialize_block_pairs(self):
        eye = torch.eye(2)
        sigma_x = torch.tensor([[0, 1], [1, 0]])
        sigma_iy = torch.tensor([[0, 1], [-1, 0]])
        sigma_z = torch.tensor([[1, 0], [0, -1]])
        blocks = torch.stack([eye, sigma_x, sigma_iy, sigma_z], dim=0)
        blocks_ids = list(range(len(blocks)))
        block_pairs = list(product(blocks_ids, blocks_ids))
        return blocks, block_pairs


    def _initialize_block_pair_idx_map(self):
        return {
            '11': 0,
            '1x': 1,
            '1y': 2,
            '1z': 3,
            'x1': 4,
            'xx': 5,
            'xy': 6,
            'xz': 7,
            'y1': 8,
            'yx': 9,
            'yy': 10,
            'yz': 11,
            'z1': 12,
            'zx': 13,
            'zy': 14,
            'zz': 15,
        }


    def _get_mlp(self, layers_num: int, input_size: int, hidden_size: int, output_size: int, final_activation: str = 'self'):
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
        if final_activation == 'self':
            layers.append(self._get_activation())
        elif final_activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        elif final_activation == 'tanh':
            layers.append(nn.Tanh())
        else:
            raise ValueError(f'Activation function: {final_activation} not implemented')
        return nn.Sequential(*layers)


    def _get_activation(self):
        if self.activation == 'relu':
            return nn.ReLU()
        elif self.activation == 'leaky_relu':
            return nn.LeakyReLU(negative_slope=0.01)
        else:
            return ValueError(f'Activation function: {self.activation} not implemented')


    def _periodic_func(self, x: torch.Tensor, i: int):
        if i % 2 == 0:
            return torch.sin(x * 2**((i // 2) / 4))
        else:
            return torch.cos(x * 2**((i // 2) / 4))


    def forward(self, x: torch.Tensor):
        block = self.naive_block_decoder(x[:, self.freq_dim:]).unsqueeze(0)
        block_expand = block.expand(self.N, -1, -1).permute((1, 2, 0)).unsqueeze(2)

        freq = self.freq_decoder(x[:, :self.freq_dim])
        freq_seq = torch.stack([self._periodic_func(self.freq_seq_constructor[i](freq), i) for i in range(self.seq_num)], dim=-1)

        tf_input = freq_seq * block_expand.squeeze(2).transpose(1, 2)
        seq = self.tf_seq_decoder(tf_input)
        seq = seq.transpose(1, 2).unsqueeze(2)

        naive_seq = self.naive_seq_decoder(x).unsqueeze(1).unsqueeze(1)

        block_seq = torch.cat([block_expand, seq, naive_seq], dim=1)
        block_seq = self.lin_mixer(block_seq).squeeze(2)

        interaction_block_seq = block_seq[:, :self.channel_num - 2]
        on_site_block_seq = block_seq[:, self.channel_num - 2:]

        H_interaction = torch.stack([self._interaction_block_generator(interaction_block_seq[:, i]) for i in range(self.channel_num - 2)], dim=1)
        num_blocks = len(self.reduced_block_pairs)
        H_on_site = torch.stack([self._on_site_block_generator(on_site_block_seq[:, :num_blocks]), self._on_site_block_generator(on_site_block_seq[:, num_blocks:])], dim=1)
        strips = torch.cat([H_interaction[:, :(self.channel_num // 2 - 1)], H_on_site, H_interaction[:, (self.channel_num // 2 - 1):]], dim=1)
        matrix = self._get_matrix_from_strips(strips)
        return matrix


    def _interaction_block_generator(self, t: torch.Tensor):
        '''
        assumes t.shape = (batch_size, seq_size)
        '''
        zz_pair = self.block_pairs[self.block_pair_idx_map['z1']]
        return self._block_generator(t, self.blocks[zz_pair[0]], self.blocks[zz_pair[1]])


    def _on_site_block_generator(self, param_vec: torch.Tensor):
        '''
        assumes param_vec.shape = (batch_size, 16, seq_size)
        '''
        assert param_vec.shape[1] == len(self.reduced_block_pairs)
        all_blocks = torch.stack([self._block_generator(param_vec[:, i, :], self.blocks[pair[0]], self.blocks[pair[1]]) for i, pair in enumerate(self.reduced_block_pairs)], dim=1)
        return torch.sum(all_blocks, dim=1)


    def _block_generator(self, x: torch.Tensor, block_a: torch.Tensor, block_b: torch.Tensor):
        '''
        assumes:
          x.shape = (batch_size, seq_size)
          block_a.shape = (2, 2)
          block_b.shape = (2, 2)
        '''
        x_broadcast = x.unsqueeze(-1).unsqueeze(-1)
        block = torch.kron(block_a.to(x.device), block_b.to(x.device))
        block_expanded = block.unsqueeze(0).unsqueeze(0).expand(x.shape[0], x.shape[1], -1, -1)
        full_block = x_broadcast*block_expanded
        return torch.cat([full_block[:, i, :, :] for i in range(full_block.shape[1])], dim=-1)


    def _get_matrix_from_strips(self, strips: torch.Tensor):
        matrix = torch.zeros((strips.shape[0], 2, self.N*self.block_size, self.N*self.block_size)).to(strips.device)
        strips_split = torch.tensor_split(strips, self.channel_num // 2, dim=1)
        for i, strip in enumerate(strips_split):
            offset = i - (len(strips_split) // 2)
            strip_off = max(0, -offset)*self.block_size
            matrix_off = abs(offset)*self.block_size
            for j in range(self.N - abs(offset)):
                idx0 =  j*self.block_size
                idx1 = (j+1)*self.block_size
                if offset >= 0:
                    matrix[:, :, idx0: idx1, idx0 + matrix_off: idx1 + matrix_off] = strip[:, :, :, idx0 + strip_off: idx1 + strip_off]
                else:
                    matrix[:, :, idx0 + matrix_off: idx1 + matrix_off, idx0: idx1] = strip[:, :, :, idx0 + strip_off: idx1 + strip_off]
        return matrix