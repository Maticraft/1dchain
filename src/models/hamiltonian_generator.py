import typing as t
from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F


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
                self.block_pairs[self.block_pair_idx_map['1y']],
            ]

        self.seq_num = 2*len(self.reduced_block_pairs) + self.channel_num - 2
        if kwargs.get('seq_num', None) is not None:
            self.seq_num = kwargs.get('seq_num')

        self.freq_decoder = self._get_mlp(self.freq_dec_depth, self.freq_dim, self.freq_dec_hidden_size, self.freq_dec_hidden_size)
        self.freq_seq_constructor = nn.ModuleList([
            self._get_mlp(self.freq_dec_depth, self.freq_dec_hidden_size, self.freq_dec_hidden_size, self.N, final_activation='sigmoid')
            for _ in range(self.seq_num)
        ])

        self.naive_block_decoder = self._get_mlp(self.block_dec_depth, self.block_dim, self.block_dec_hidden_size, self.seq_num)

        self.naive_seq_decoder = self._get_mlp(self.seq_dec_depth, self.freq_dim+self.block_dim, self.seq_dec_hidden_size, self.N)

        self.tf_seq_decoder_layer = nn.TransformerEncoderLayer(d_model=self.seq_num, nhead=2, dim_feedforward=128, batch_first=True)
        self.tf_seq_decoder = nn.TransformerEncoder(self.tf_seq_decoder_layer, num_layers=1)

        output_channels_num = 2*len(self.reduced_block_pairs) + self.channel_num - 2 # same as seq_num?
        self.lin_mixer = nn.Conv2d(2*self.seq_num + 1, output_channels_num, kernel_size=1, stride=1)


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
                layers.append(nn.BatchNorm1d(hidden_size))  # batchnorm location fixed
            elif i == layers_num - 1:
                layers.append(nn.Linear(hidden_size, output_size))
                layers.append(nn.BatchNorm1d(hidden_size))  # batchnorm location fixed
            else:
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
        z1_pair = self.block_pairs[self.block_pair_idx_map['z1']]
        return self._block_generator(t, self.blocks[z1_pair[0]], self.blocks[z1_pair[1]])


    def _on_site_block_generator(self, param_vec: torch.Tensor):
        '''
        assumes param_vec.shape = (batch_size, blocks_num, seq_size)
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
        returns:
          torch.Tensor of shape (batch_size, 4, 4*seq_size)
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
            matrix_off = abs(offset)*self.block_size
            for j in range(self.N):
                idx0 =  j*self.block_size
                idx1 = (j+1)*self.block_size
                if offset >= 0:
                    matrix_idx0 = (idx0 + matrix_off) % (self.N*self.block_size)
                    matrix_idx1 = matrix_idx0 + self.block_size
                else:
                    matrix_idx0 = (idx0 - matrix_off) % (self.N*self.block_size)
                    matrix_idx1 = matrix_idx0 + self.block_size
                matrix[:, :, idx0: idx1, matrix_idx0: matrix_idx1] = strip[:, :, :, idx0: idx1]
        return matrix
    

class HamiltonianGeneratorV2(HamiltonianGenerator):
    def __init__(self, representation_dim: t.Union[int, t.Tuple[int, int]], output_size: t.Tuple[int, int, int], **kwargs: t.Dict[str, t.Any]):
        super().__init__(representation_dim, output_size, **kwargs)
        self.varying_potential = kwargs.get('varying_potential', False)
        self.varying_delta = kwargs.get('varying_delta', False)
        num_on_site_varying_params = 2
        if self.varying_potential:
            num_on_site_varying_params += 1
        if self.varying_delta:
            num_on_site_varying_params += 1
        num_site_varying_params = num_on_site_varying_params + self.channel_num - 2
        self.varying_block_mixer = nn.Conv1d(self.seq_num + 1, num_site_varying_params, kernel_size=1, stride=1)
        num_site_constant_params = 2
        self.constant_block_mixer = nn.Conv1d(self.seq_num, num_site_constant_params, kernel_size=1, stride=1)
        self.strip_longitude_interaction_conv = nn.Conv1d(self.channel_num - 2, 2, kernel_size=2, stride=1, dilation=2, bias=False)
        self.strip_latitude_interaction_conv = nn.Conv1d(self.channel_num - 2, 2, kernel_size=2, stride=2, dilation=1, bias=False)
        self.smoothing = kwargs.get('smoothing', False)

    def forward(self, x: torch.Tensor):
        block = self.naive_block_decoder(x[:, self.freq_dim:]).unsqueeze(0)
        block_expand = block.expand(self.N, -1, -1).transpose(0, 1) # site constant blocks

        freq = self.freq_decoder(x[:, :self.freq_dim])
        freq_seq = torch.stack([self._periodic_func(self.freq_seq_constructor[i](freq), i) for i in range(self.seq_num)], dim=-1)

        tf_input = freq_seq * block_expand
        seq = self.tf_seq_decoder(tf_input)
        seq = seq.transpose(1, 2) #.unsqueeze(2)

        naive_seq = self.naive_seq_decoder(x).unsqueeze(1) #.unsqueeze(1)

        block_seq = torch.cat([seq, naive_seq], dim=1)
        block_seq = self.varying_block_mixer(block_seq) # site varying blocks

        interaction_block_seq = block_seq[:, :self.channel_num - 2]
        latitude_interactions = self.strip_latitude_interaction_conv(interaction_block_seq) # of shape (batch_size, 2, N/2)
        longitude_interactions = self.strip_longitude_interaction_conv(F.pad(interaction_block_seq, pad=(0, 2), mode='circular')) # of shape (batch_size, 2, N)
        H_interaction_lower, H_interaction_upper = self._construct_interactions(latitude_interactions, longitude_interactions)

        on_site_block_seq = block_seq[:, self.channel_num - 2:]
        # smooth on site blocks by calculating the average of N neighboring sites
        on_site_block_seq = self._calculate_running_mean(on_site_block_seq) if self.smoothing else on_site_block_seq

        block_expand = block_expand.transpose(1, 2)
        constant_blocks = self.constant_block_mixer(block_expand)
        
        real_blocks = self._on_site_real_block_generator(on_site_block_seq, constant_blocks)
        imaginary_blocks = self._on_site_imag_block_generator(on_site_block_seq, constant_blocks)
        H_on_site = torch.stack([real_blocks, imaginary_blocks], dim=1)

        strips = torch.cat([H_interaction_lower, H_on_site, H_interaction_upper], dim=1)
        matrix = self._get_matrix_from_strips(strips)
        return matrix
    

    def _construct_interactions(self, latitude_interactions: torch.Tensor, longitude_interactions: torch.Tensor):
        '''
        latitude_interactions.shape = (batch_size, 2, N/2)
        longitude_interactions.shape = (batch_size, 2, N)
        '''
        # insert 0s every 2nd element (in last dimension) in latitude_interactions
        latitude_interactions = torch.flatten(torch.stack((latitude_interactions, torch.zeros_like(latitude_interactions)), dim=-1), start_dim=-2)
        interactions = torch.cat((latitude_interactions, longitude_interactions), dim=1)
        H_interaction = torch.stack([self._interaction_block_generator(interactions[:, i]) for i in range(4)], dim=1)
        
        H_latitude_interactions_lower = torch.cat([H_interaction[:, :2, :, (self.N-1)*self.block_size:], H_interaction[:, :2, :, :(self.N-1)*self.block_size]], dim=-1)
        H_longitude_interactions_lower = torch.cat([H_interaction[:, 2:, :, (self.N-2)*self.block_size:], H_interaction[:, 2:, :, :(self.N-2)*self.block_size]], dim=-1)
        H_interaction_lower = torch.cat([H_longitude_interactions_lower, H_latitude_interactions_lower], dim=1) # order of latitude and longitude must be inverted in lower part of hamiltonian
        H_interaction_lower[:, 1::2] *= -1 # conjugation of imaginary part

        return H_interaction_lower, H_interaction
    

    def _calculate_running_mean(self, x: torch.Tensor):
        '''
        x.shape = (batch_size, num_blocks, seq_size)
        '''
        neighbors_to_average = [x[..., list(filter(lambda k: k>=0 and k<x.shape[-1], [i-2, i, i+2, i+1-2*(i%2)]))] for i in range(x.shape[-1])]
        x_averaged = torch.stack([torch.mean(neighbors, dim=-1) for neighbors in neighbors_to_average], dim=-1)
        return x_averaged
    

    def _on_site_real_block_generator(self, varying_blocks: torch.Tensor, constant_blocks: torch.Tensor):
        z1_pair = self.block_pairs[self.block_pair_idx_map['z1']] # for chemical potential
        zx_pair = self.block_pairs[self.block_pair_idx_map['zx']] # for spin real part
        yy_pair = self.block_pairs[self.block_pair_idx_map['yy']] # for superconductivity

        chemical_potential_block = self._block_generator(constant_blocks[:, 0, :], self.blocks[z1_pair[0]], self.blocks[z1_pair[1]])
        if self.varying_potential:
            varying_chemical_potential_block = self._block_generator(varying_blocks[:, 2, :], self.blocks[z1_pair[0]], self.blocks[z1_pair[1]])
            chemical_potential_block += varying_chemical_potential_block
        delta_block = self._block_generator(constant_blocks[:, 1, :], self.blocks[yy_pair[0]], self.blocks[yy_pair[1]])
        if self.varying_delta:
            varying_delta_block = self._block_generator(varying_blocks[:, 3, :], self.blocks[yy_pair[0]], self.blocks[yy_pair[1]])
            delta_block += varying_delta_block
        spin_block = self._block_generator(varying_blocks[:, 0, :], self.blocks[zx_pair[0]], self.blocks[zx_pair[1]])

        all_blocks = torch.stack([chemical_potential_block, delta_block, spin_block], dim=1)
        return torch.sum(all_blocks, dim=1)
    

    def _on_site_imag_block_generator(self, varying_blocks: torch.Tensor, constant_blocks: torch.Tensor):
        zy_pair = self.block_pairs[self.block_pair_idx_map['1y']] # for spin imaginary part
        spin_block = self._block_generator(varying_blocks[:, 1, :], self.blocks[zy_pair[0]], self.blocks[zy_pair[1]])
        return spin_block
    

class QuantumDotsHamiltonianGenerator(HamiltonianGeneratorV2):
    def __init__(self, representation_dim: t.Union[int, t.Tuple[int, int]], output_size: t.Tuple[int, int, int], **kwargs: t.Dict[str, t.Any]):
        super().__init__(representation_dim, output_size, **kwargs)
        # assuming all params can vary between sites (quantum dots and levels)
        self.num_levels = (self.channel_num // 2 - 1) // 2
        num_on_site_varying_params = 4
        num_site_varying_params = num_on_site_varying_params + self.channel_num - 2
        self.varying_block_mixer = nn.Conv1d(self.seq_num + 1, num_site_varying_params, kernel_size=1, stride=1)
        num_site_constant_params = 4
        self.constant_block_mixer = nn.Conv1d(self.seq_num, num_site_constant_params, kernel_size=1, stride=1)
        self.num_interaction_params = 4
        self.strip_longitude_interaction_conv = nn.Conv1d(self.channel_num - 2, self.num_interaction_params, kernel_size=2, stride=1, dilation=self.num_levels, bias=False)
        # in case of quantum dots latitute interactions are interactions between different levels (assumed only neighboring levels interact with each other)
        self.strip_latitude_interaction_conv = nn.Conv1d(self.channel_num - 2, self.num_interaction_params*(self.num_levels - 1), kernel_size=self.num_levels, stride=self.num_levels, dilation=1, bias=False)


    def _construct_interactions(self, latitude_interactions: torch.Tensor, longitude_interactions: torch.Tensor):
        '''
        latitude_interactions.shape = (batch_size, num_interaction_params*(num_levels - 1), N/num_levels)
        longitude_interactions.shape = (batch_size, num_interaction_params, N)
        '''
        latitude_interactions = torch.cat((latitude_interactions, torch.zeros((latitude_interactions.shape[0], self.num_interaction_params, self.N//self.num_levels), device=latitude_interactions.device)), dim=1)
        latitude_interactions = latitude_interactions.view(-1, self.num_interaction_params, self.N)
        num_interaction_params_per_block = self.num_interaction_params // 2 # separate for real and imaginary part

        # since only nearest levels interactions are considers we must to fill the missing interactions with 0s
        if self.num_levels == 1:
            interactions = longitude_interactions
            H_interaction = torch.stack([self._interaction_block_generator(interactions[:, num_interaction_params_per_block*i:num_interaction_params_per_block*(i+1)], is_real=i%2==0) for i in range(interactions.shape[1] // num_interaction_params_per_block)], dim=1)
            H_longitude_lower = torch.cat([H_interaction[:, :, :, (self.N-self.num_levels)*self.block_size:], H_interaction[:, :, :, :(self.N-self.num_levels)*self.block_size]], dim=-1)
            H_interaction_lower[:, 1::2] *= -1 # conjugation of imaginary part
            return H_longitude_lower, H_interaction

        skipped_levels = self.num_levels - 2
        missing_latitude_interactions = torch.zeros((latitude_interactions.shape[0], skipped_levels*self.num_interaction_params, self.N), device=latitude_interactions.device)

        interactions = torch.cat((latitude_interactions, missing_latitude_interactions, longitude_interactions), dim=1) # shape = (batch_size, 2*num_interaction_params, N)
        H_interaction = torch.stack([self._interaction_block_generator(interactions[:, num_interaction_params_per_block*i:num_interaction_params_per_block*(i+1)], is_real=i%2==0) for i in range(interactions.shape[1] // num_interaction_params_per_block)], dim=1)
        
        H_latitude_interactions_lower = torch.cat([H_interaction[:, :2, :, (self.N-1)*self.block_size:], H_interaction[:, :2, :, :(self.N-1)*self.block_size]], dim=-1)
        # dummy interactions for missing levels can be considered either as latitute or longitude (they are neglecitble due to their 0 value)
        H_longitude_interactions_lower = torch.cat([H_interaction[:, (skipped_levels + 1)*2:, :, (self.N-self.num_levels)*self.block_size:], H_interaction[:, (skipped_levels + 1)*2:, :, :(self.N-self.num_levels)*self.block_size]], dim=-1)
        H_missing_interactions = H_interaction[:, 2:2*(skipped_levels + 1), :, :]
        H_interaction_lower = torch.cat([H_longitude_interactions_lower, H_missing_interactions, H_latitude_interactions_lower], dim=1) # order of latitude and longitude must be inverted in lower part of hamiltonian
        H_interaction_lower[:, 1::2] *= -1 # conjugation of imaginary part
        
        # switch the order of the interactions in the lower part of the hamiltonian
        right_upper_coeff = H_interaction_lower[:, :, 0, 1::4].clone()
        left_upper_coeff = H_interaction_lower[:, :, 1, 0::4].clone()
        H_interaction_lower[:, :, 0, 1::4] = left_upper_coeff
        H_interaction_lower[:, :, 1, 0::4] = right_upper_coeff

        right_lower_coeff = H_interaction_lower[:, :, 2, 3::4].clone()
        left_lower_coeff = H_interaction_lower[:, :, 3, 2::4].clone()
        H_interaction_lower[:, :, 2, 3::4] = left_lower_coeff
        H_interaction_lower[:, :, 3, 2::4] = right_lower_coeff

        return H_interaction_lower, H_interaction
    

    def _interaction_block_generator(self, t: torch.Tensor, is_real: bool):
        '''
        assumes t.shape = (batch_size, seq_size)
        '''
        if is_real:
            z1_pair = self.block_pairs[self.block_pair_idx_map['z1']]
            _1y_pair = self.block_pairs[self.block_pair_idx_map['1y']]
            z1_block = self._block_generator(t[:, 0], self.blocks[z1_pair[0]], self.blocks[z1_pair[1]])
            _1y_block = self._block_generator(t[:, 1], self.blocks[_1y_pair[0]], self.blocks[_1y_pair[1]])
            return z1_block + _1y_block
        _1z_pair = self.block_pairs[self.block_pair_idx_map['1z']]
        _1x_pair = self.block_pairs[self.block_pair_idx_map['1x']]
        _1z_block = self._block_generator(t[:, 0], self.blocks[_1z_pair[0]], self.blocks[_1z_pair[1]])
        _1x_block = self._block_generator(t[:, 1], self.blocks[_1x_pair[0]], self.blocks[_1x_pair[1]])
        return _1z_block + _1x_block


    def _on_site_real_block_generator(self, varying_blocks: torch.Tensor, constant_blocks: torch.Tensor):
        z1_pair = self.block_pairs[self.block_pair_idx_map['z1']]
        zz_pair = self.block_pairs[self.block_pair_idx_map['zz']]
        yy_pair = self.block_pairs[self.block_pair_idx_map['yy']]

        z1_block = self._block_generator(constant_blocks[:, 0, :], self.blocks[z1_pair[0]], self.blocks[z1_pair[1]])
        z1_block += self._block_generator(varying_blocks[:, 0, :], self.blocks[z1_pair[0]], self.blocks[z1_pair[1]])
        
        zz_block = self._block_generator(constant_blocks[:, 1, :], self.blocks[zz_pair[0]], self.blocks[zz_pair[1]])
        zz_block += self._block_generator(varying_blocks[:, 1, :], self.blocks[zz_pair[0]], self.blocks[zz_pair[1]])
        
        yy_block = self._block_generator(constant_blocks[:, 2, :], self.blocks[yy_pair[0]], self.blocks[yy_pair[1]])
        yy_block += self._block_generator(varying_blocks[:, 2, :], self.blocks[yy_pair[0]], self.blocks[yy_pair[1]])
        return z1_block + zz_block + yy_block


    def _on_site_imag_block_generator(self, varying_blocks: torch.Tensor, constant_blocks: torch.Tensor):
        xy_pair = self.block_pairs[self.block_pair_idx_map['xy']]
        xy_block = self._block_generator(constant_blocks[:, 3, :], self.blocks[xy_pair[0]], self.blocks[xy_pair[1]])
        xy_block += self._block_generator(varying_blocks[:, 3, :], self.blocks[xy_pair[0]], self.blocks[xy_pair[1]])
        return xy_block
