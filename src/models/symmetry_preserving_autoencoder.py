import typing as t
from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base_models import MLP
from src.models.hamiltonian_matrix_utils import get_strip, get_matrix_from_strips
from src.models.hamiltonian_block_handlers import BlockConstructor


class DistributionPreservingEncoder(nn.Module):
    def __init__(self, input_size: t.Tuple[int, int, int], representation_dim: t.Union[int, t.Tuple[int, int]], **kwargs: t.Dict[str, t.Any]):
        super(DistributionPreservingEncoder, self).__init__()
        self.channel_num = input_size[0]
        self.N = input_size[1]
        self.block_size = input_size[2]
        if type(representation_dim) == int:
            self.freq_dim = representation_dim // 2
            self.block_dim = representation_dim // 2
        else:
            self.freq_dim = representation_dim[0]
            self.block_dim = representation_dim[1]

        self.stride = self.block_size
        self.dilation = 1
        self.site_size = (1, self.N)

        self.kernel_num = kwargs.get('kernel_num', 32)
        self.kernel_size = kwargs.get('kernel_size', self.block_size)

        self.freq_enc_depth = kwargs.get('freq_enc_depth', 2)
        self.freq_enc_hidden_size = kwargs.get('freq_enc_hidden_size', 64)

        self.block_enc_depth = kwargs.get('block_enc_depth', 2)
        self.block_enc_hidden_size = kwargs.get('block_enc_hidden_size', 64)

        self.activation = kwargs.get('activation', 'relu')
        self.padding_mode = kwargs.get('padding_mode', 'zeros')

        self.strip_len = self._get_convs_output_size(1)

        self.conv = self._get_conv_block()

        self.simple_parser = nn.ModuleList([
            MLP(self.block_enc_depth, self.strip_len, self.block_enc_hidden_size, 1, self.activation)
            for _ in range(self.kernel_num)
        ])
        self.simple_encoder = nn.Sequential(
            nn.Linear(self.kernel_num, self.block_dim),
            self._get_activation(),
        )

        self.tf_block_enc_layer = nn.TransformerEncoderLayer(d_model=self.kernel_num, nhead=2, dim_feedforward=128, batch_first=True)
        self.tf_block_encoder = nn.TransformerEncoder(self.tf_block_enc_layer, num_layers=1)
        self.block_parser = nn.ModuleList([
            MLP(self.block_enc_depth, self.strip_len, self.block_enc_hidden_size, 1, self.activation)
            for _ in range(self.kernel_num)
        ])
        self.block_encoder = nn.Sequential(
            nn.Linear(self.kernel_num, self.block_dim),
            self._get_activation(),
        )

        self.fft_parser = nn.ModuleList([
            MLP(self.freq_enc_depth, self.strip_len, self.freq_enc_hidden_size, 1, self.activation)
            for _ in range(self.kernel_num)
        ])
        self.fft_encoder = nn.Sequential(
            nn.Linear(self.kernel_num, self.freq_dim),
            self._get_activation(),
        )

        self.tf_freq_enc_layer = nn.TransformerEncoderLayer(d_model=self.kernel_num, nhead=2, dim_feedforward=128, batch_first=True)
        self.tf_freq_encoder = nn.TransformerEncoder(self.tf_freq_enc_layer, num_layers=1)
        self.freq_parser = nn.ModuleList([
            MLP(self.freq_enc_depth, self.strip_len, self.freq_enc_hidden_size, 1, self.activation)
            for _ in range(self.kernel_num)
        ])
        self.freq_encoder = nn.Sequential(
            nn.Linear(self.kernel_num, self.freq_dim),
            self._get_activation(),
        )

        self.block_lin_encoder = nn.Linear(2*self.block_dim, self.block_dim)
        self.freq_lin_encoder = nn.Linear(2*self.freq_dim, self.freq_dim)


    def _get_activation(self):
        if self.activation == 'relu':
            return nn.ReLU()
        elif self.activation == 'leaky_relu':
            return nn.LeakyReLU(negative_slope=0.01)
        else:
            return ValueError(f'Activation function: {self.activation} not implemented')


    def _get_conv_block(self):
        padding = (self.kernel_size - self.block_size) // 2
        return nn.Sequential(
            nn.Conv2d(self.channel_num, self.kernel_num, kernel_size=(self.block_size, self.kernel_size), stride=self.stride, dilation=self.dilation, padding=(0, padding), padding_mode=self.padding_mode),
            nn.BatchNorm2d(self.kernel_num),
            self._get_activation(),
        )


    def _get_convs_output_size(self, dim):
        return (self.site_size[dim]*self.block_size - self.dilation*(self.block_size-1) - 1) // self.stride + 1


    def forward(self, x: torch.Tensor):
        strip_bound = ((self.channel_num // 2) - 1) // 2
        x = torch.cat([get_strip(x, i, self.N, 'hamiltonian', self.block_size) for i in range(-strip_bound, strip_bound + 1)], dim=1)
        x = self.conv(x)
        seq_strips = x.view(-1, self.kernel_num, self.strip_len)

        simple_out = torch.cat([self.simple_parser[i](seq_strips[:, i, :]) for i in range(self.kernel_num)], dim=-1)
        simple_out = self.simple_encoder(simple_out)

        block_strips = seq_strips.transpose(1, 2)
        block_seq = self.tf_block_encoder(block_strips)
        block_out = torch.cat([self.block_parser[i](block_seq[:, :, i]) for i in range(self.kernel_num)], dim=-1)
        block_out = self.block_encoder(block_out)

        block_enc = self.block_lin_encoder(torch.cat([simple_out, block_out], dim=-1))

        complex_strips = torch.stack([torch.complex(seq_strips[:, i, :], seq_strips[:, self.kernel_num // 2 + i, :]) for i in range(self.kernel_num // 2)], dim=1)
        fft_strips = torch.fft.fft(complex_strips, dim=-1)
        fft_strips = torch.cat([fft_strips.real, fft_strips.imag], dim=1)

        fft_out = torch.cat([self.fft_parser[i](fft_strips[:, i, :]) for i in range(self.kernel_num)], dim=-1)
        fft_out = self.fft_encoder(fft_out)

        fft_strips = fft_strips.transpose(1, 2)
        freq_seq = self.tf_freq_encoder(fft_strips)
        freq_out = torch.cat([self.freq_parser[i](freq_seq[:, :, i]) for i in range(self.kernel_num)], dim=-1)
        freq_out = self.freq_encoder(freq_out)

        freq_enc = self.freq_lin_encoder(torch.cat([fft_out, freq_out], dim=-1))

        return torch.cat([freq_enc, block_enc], dim=-1)


class DistributionPreservingHamiltonianGenerator(nn.Module):
    def __init__(self, representation_dim: int, output_size: t.Tuple[int, int, int], **kwargs: t.Dict[str, t.Any]):
        super(DistributionPreservingHamiltonianGenerator, self).__init__()
        self.channel_num = output_size[0]
        assert self.channel_num % 2 == 0, 'Channel number must be even'
        self.N = output_size[1]
        self.block_size = output_size[2]

        self.on_site_real_block_pairs = kwargs.get('on_site_real_block_pairs', ['z1', 'zx', 'zz', 'iyiy'])
        self.on_site_imag_block_pairs = kwargs.get('on_site_imag_block_pairs', ['1iy', 'xiy'])
        self.interaction_real_block_pairs = kwargs.get('interaction_real_block_pairs', ['z1', '1y'])
        self.interaction_imag_block_pairs = kwargs.get('interaction_imag_block_pairs', ['z1', '1z', '1x']) # why for ladder z1 is used while for qdh it is 1z?
        
        self.total_on_site_params = len(self.on_site_real_block_pairs) + len(self.on_site_imag_block_pairs)
        self.num_independent_interaction_strips = (self.channel_num // 2 - 1) // 2
        self.interaction_params_per_strip = len(self.interaction_real_block_pairs) + len(self.interaction_imag_block_pairs)
        self.total_interaction_params = self.interaction_params_per_strip * self.num_independent_interaction_strips
        self.total_params = self.total_on_site_params + self.total_interaction_params
        self.total_distribution_params = 2*self.total_params # for mean and std
        self.encoded_data_dim = representation_dim - self.total_distribution_params

        self.dec_depth = kwargs.get('dec_depth', 4)
        self.dec_hidden_size = kwargs.get('dec_hidden_size', 128)

        self.seq_dec_depth = kwargs.get('seq_dec_depth', 4)
        self.seq_dec_hidden_size = kwargs.get('seq_dec_hidden_size', 128)
        self.seq_channels_num = kwargs.get('seq_channels_num', 64)

        self.activation = kwargs.get('activation', 'relu')

        self.amplitudes_decoder = nn.ModuleList([
            MLP(self.seq_dec_depth, self.encoded_data_dim, self.seq_dec_hidden_size, self.N, self.activation, final_activation='sigmoid')
            for _ in range(self.seq_channels_num)
        ])
        self.freq_decoder = MLP(self.dec_depth, self.encoded_data_dim, self.dec_hidden_size, self.dec_hidden_size, self.activation)
        self.freq_seq_constructor = nn.ModuleList([
            MLP(self.dec_depth, self.dec_hidden_size, self.dec_hidden_size, self.N, self.activation, final_activation='sigmoid')
            for _ in range(self.seq_channels_num)
        ])

        self.tf_seq_decoder_layer = nn.TransformerEncoderLayer(d_model=self.seq_channels_num, nhead=2, dim_feedforward=128, batch_first=True) # consider different transformer for each block parameter
        self.tf_seq_decoder = nn.TransformerEncoder(self.tf_seq_decoder_layer, num_layers=1)

        self.on_site_block_mixer = nn.Conv1d(self.seq_channels_num, self.total_on_site_params, kernel_size=1, stride=1)
        self.interaction_block_mixers = nn.ModuleList([
            nn.Conv1d(self.seq_channels_num, self.total_interaction_params, kernel_size=2, stride=1, dilation=interaction_range, bias=False)
            for interaction_range in range(1, self.num_independent_interaction_strips + 1)
        ])
        self.allow_periodic = kwargs.get('allow_periodic', False)
        self.interlevels_interactions = kwargs.get('interlevels_interactions', False)
        self.smoothing = kwargs.get('smoothing', False)


    def forward(self, x: torch.Tensor):
        distribution_data = x[:, :self.total_distribution_params]
        encoded_data = x[:, self.total_distribution_params:]

        on_site_seq, interactions_seq = self._propagate_neural_network(encoded_data)
        
        on_site_mean = distribution_data[:, :self.total_on_site_params]
        on_site_std = distribution_data[:, self.total_params: self.total_params + self.total_on_site_params]
        interactions_mean = distribution_data[:, self.total_on_site_params: self.total_params].view(-1, self.num_independent_interaction_strips, self.interaction_params_per_strip)
        interactions_std = distribution_data[:, self.total_params + self.total_on_site_params:].view(-1, self.num_independent_interaction_strips, self.interaction_params_per_strip)

        denormalized_on_site_seq = self._denormalize_seq(on_site_seq, on_site_mean, on_site_std)
        denormalized_interactions_seq = self._denormalize_seq(interactions_seq, interactions_mean, interactions_std)

        interaction_strips_lower, interaction_strips_upper = self._construct_interactions(denormalized_interactions_seq)

        on_site_strips = self._construct_on_site_blocks(denormalized_on_site_seq)

        strips = torch.cat([interaction_strips_lower, on_site_strips, interaction_strips_upper], dim=1)
        matrix = get_matrix_from_strips(strips, self.N, self.block_size)
        return matrix


    def _propagate_neural_network(self, x: torch.Tensor):
        amplitudes = torch.stack([amp_dec(x) for amp_dec in self.amplitudes_decoder], dim=-1) # of shape (batch_size, N, seq_channels_num)
        freq = self.freq_decoder(x)
        freq_seq = torch.stack([self._periodic_func(self.freq_seq_constructor[i](freq), i) for i in range(self.seq_channels_num)], dim=-1) # of shape (batch_size, N, seq_channels_num)

        tf_input = freq_seq * amplitudes
        seq = self.tf_seq_decoder(tf_input)
        seq = seq.transpose(1, 2) # of shape (batch_size, seq_channels_num, N)

        on_site_seq = self.on_site_block_mixer(seq) # blocks of shape (batch_size, total_on_site_params, N)
        interactions_seq = torch.stack([
            interaction_block_mixer(F.pad(seq, pad=(0, i + 1), mode='circular')) for i, interaction_block_mixer in enumerate(self.interaction_block_mixers)
        ], dim=1) # of shape (batch_size, num_independent_interaction_strips, interaction_params_per_strip, N)
        return on_site_seq, interactions_seq

    def _periodic_func(self, x: torch.Tensor, i: int):
        if i % 2 == 0:
            return torch.sin(x * 2**((i // 2) / 4))
        else:
            return torch.cos(x * 2**((i // 2) / 4))
        

    def _denormalize_seq(self, seq: torch.Tensor, seq_mean: torch.Tensor, seq_std: torch.Tensor):
        '''
        seq.shape = (..., n_params, N)
        seq_mean.shape = (..., n_params)
        seq_std.shape = (..., n_params)
        '''
        return seq * seq_std.unsqueeze(-1) + seq_mean.unsqueeze(-1)


    def _construct_interactions(self, interactions: torch.Tensor):
        '''
        interactions.shape = (batch_size, num_independent_interaction_strips, interaction_params_per_strip, N)
        '''
        interaction_strips_upper = []
        interaction_strips_lower = []
        for interaction_strip_idx in range(interactions.shape[1]):
            interaction_strip_real = self._interaction_real_block_generator(interactions[:, interaction_strip_idx])
            interaction_strip_imag = self._interaction_imag_block_generator(interactions[:, interaction_strip_idx])
            upper_interaction_strip = torch.stack([interaction_strip_real, interaction_strip_imag], dim=1) # of shape (batch_size, 2, 4, 4*N)
            periodic_interactions_offset = (interaction_strip_idx + 1) * self.block_size
            lower_interaction_strip = self._generate_lower_strip_from_upper_strip(upper_interaction_strip, periodic_interactions_offset)
            interaction_strips_upper.append(upper_interaction_strip)
            interaction_strips_lower.append(lower_interaction_strip)
        interaction_strips_upper = torch.cat(interaction_strips_upper, dim=1)
        interaction_strips_lower = torch.cat(interaction_strips_lower[::-1], dim=1)
        return interaction_strips_lower, interaction_strips_upper
    
    @staticmethod
    def _generate_lower_strip_from_upper_strip(upper_strip: torch.Tensor, periodic_interactions_offset: int):
        # shift the periodic interactions from the right side of the upper strip to the left side of the lower strip
        lower_strip = torch.cat([upper_strip[:, :, :, -periodic_interactions_offset:], upper_strip[:, :, :, :-periodic_interactions_offset]], dim=-1)
        lower_strip[:, 1::2] *= -1 # conjugation of imaginary part
        # transpose the strip (switch the order of the non-diagonal interactions in the lower strip)
        right_upper_coeff = upper_strip[:, :, 0, 1::4].clone()
        left_upper_coeff = upper_strip[:, :, 1, 0::4].clone()
        lower_strip[:, :, 0, 1::4] = left_upper_coeff
        lower_strip[:, :, 1, 0::4] = right_upper_coeff

        right_lower_coeff = upper_strip[:, :, 2, 3::4].clone()
        left_lower_coeff = upper_strip[:, :, 3, 2::4].clone()
        lower_strip[:, :, 2, 3::4] = left_lower_coeff
        lower_strip[:, :, 3, 2::4] = right_lower_coeff
        return lower_strip
    
    def _interaction_real_block_generator(self, interations_blocks: torch.Tensor):
        '''
        assumes interations_blocks.shape = (batch_size, interaction_params, seq_size)
        '''
        end_indx = len(self.interaction_real_block_pairs)
        blocks = interations_blocks[:, :end_indx]
        return BlockConstructor.generate_block_sequence(blocks.transpose(0, 1), ['z1', '1y'])

    def _interaction_imag_block_generator(self, interations_blocks: torch.Tensor):
        '''
        assumes interations_blocks.shape = (batch_size, interaction_params, seq_size)
        '''
        start_indx = len(self.interaction_real_block_pairs)
        blocks = interations_blocks[:, start_indx:]
        return BlockConstructor.generate_block_sequence(blocks.transpose(0, 1), ['1z', '1x'])


    def _construct_on_site_blocks(self, on_site_blocks: torch.Tensor):
        '''
        assumes on_site_blocks.shape = (batch_size, total_on_site_params, seq_size)
        '''
        real_blocks = self._on_site_real_block_generator(on_site_blocks)
        imaginary_blocks = self._on_site_imag_block_generator(on_site_blocks)
        return torch.stack([real_blocks, imaginary_blocks], dim=1)

    def _on_site_real_block_generator(self, on_site_blocks: torch.Tensor):
        '''
        assumes on_site_blocks.shape = (batch_size, total_on_site_params, seq_size)
        '''
        end_idx = len(self.on_site_real_block_pairs)
        blocks = on_site_blocks[:, :end_idx]
        return BlockConstructor.generate_block_sequence(blocks.transpose(0, 1), self.on_site_real_block_pairs)

    def _on_site_imag_block_generator(self, on_site_blocks: torch.Tensor):
        '''
        assumes on_site_blocks.shape = (batch_size, total_on_site_params, seq_size)
        '''
        start_idx = len(self.on_site_real_block_pairs)
        blocks = on_site_blocks[:, start_idx:]
        return BlockConstructor.generate_block_sequence(blocks.unsqueeze(0), self.on_site_imag_block_pairs)
