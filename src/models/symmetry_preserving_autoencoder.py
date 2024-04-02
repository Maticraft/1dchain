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
        self.total_interaction_params = (len(self.interaction_real_block_pairs) + len(self.interaction_imag_block_pairs)) * (self.channel_num // 2 - 1)
        self.total_params = self.total_on_site_params + self.total_interaction_params
        self.total_distribution_params = 2*self.total_params # for mean and std
        self.encoded_data_dim = representation_dim - self.total_distribution_params

        self.dec_depth = kwargs.get('dec_depth', 4)
        self.dec_hidden_size = kwargs.get('dec_hidden_size', 128)

        self.seq_dec_depth = kwargs.get('seq_dec_depth', 4)
        self.seq_dec_hidden_size = kwargs.get('seq_dec_hidden_size', 128)
        self.seq_channels_num = kwargs.get('seq_channels_num', None)

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
        self.interaction_block_mixer = nn.Conv1d(self.seq_channels_num, self.total_interaction_params, kernel_size=2, stride=1, dilation=1, bias=False)
        self.allow_periodic = kwargs.get('allow_periodic', False)
        self.interlevels_interactions = kwargs.get('interlevels_interactions', False)
        self.smoothing = kwargs.get('smoothing', False)


    def forward(self, x: torch.Tensor):
        distribution_data = x[:, :self.total_distribution_params]
        encoded_data = x[:, self.total_distribution_params:]

        amplitudes = torch.stack([amp_dec(encoded_data) for amp_dec in self.amplitudes_decoder], dim=-1) # of shape (batch_size, N, seq_channels_num)
        freq = self.freq_decoder(encoded_data)
        freq_seq = torch.stack([self._periodic_func(self.freq_seq_constructor[i](freq), i) for i in range(self.seq_channels_num)], dim=-1) # of shape (batch_size, N, seq_channels_num)

        tf_input = freq_seq * amplitudes
        seq = self.tf_seq_decoder(tf_input)
        seq = seq.transpose(1, 2) # of shape (batch_size, seq_channels_num, N)

        on_site_seq = self.on_site_block_mixer(seq) # blocks of shape (batch_size, total_params, N)
        interactions_seq = self.interaction_block_mixer(F.pad(seq, pad=(0, 1), mode='circular')) # of shape (batch_size, total_interaction_params, N)

        # TODO denormalize interactions_seq
        H_interaction_lower, H_interaction_upper = self._construct_interactions(interactions_seq)

        # TODO denormalize on_site_block_seq
        real_blocks = self._on_site_real_block_generator(on_site_seq)
        imaginary_blocks = self._on_site_imag_block_generator(on_site_seq)
        H_on_site = torch.stack([real_blocks, imaginary_blocks], dim=1)

        strips = torch.cat([H_interaction_lower, H_on_site, H_interaction_upper], dim=1)
        matrix = get_matrix_from_strips(strips, self.N, self.block_size)
        return matrix
    

    def _periodic_func(self, x: torch.Tensor, i: int):
        if i % 2 == 0:
            return torch.sin(x * 2**((i // 2) / 4))
        else:
            return torch.cos(x * 2**((i // 2) / 4))
    

    def _construct_interactions(self, interactions: torch.Tensor):
        '''
        interactions.shape = (batch_size, total_interaction_params, N)
        '''
        latitude_interactions = torch.cat((latitude_interactions, torch.zeros((latitude_interactions.shape[0], self.num_interaction_params, self.N//self.num_levels), device=latitude_interactions.device)), dim=1)
        latitude_interactions = latitude_interactions.view(-1, self.num_interaction_params, self.N)
        num_interaction_params_per_block = self.num_interaction_params // 2 # separate for real and imaginary part

        # mock longitude interactions at periodic boundaries to 0s
        if not self.allow_periodic:
            longitude_interactions[:, :, (self.N - 2):] = 0
        if not self.interlevels_interactions:
            latitude_interactions.fill_(0)

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
            return BlockConstructor.generate_block_sequence(t.transpose(0, 1), ['z1', '1y'])
        return BlockConstructor.generate_block_sequence(t.transpose(0, 1), ['1z', '1x'])

    def _on_site_real_block_generator(self, on_site_blocks: torch.Tensor):
        end_idx = len(self.on_site_real_block_pairs)
        blocks = on_site_blocks[:, :end_idx]
        return BlockConstructor.generate_block_sequence(blocks.transpose(0, 1), self.on_site_real_block_pairs)

    def _on_site_imag_block_generator(self, on_site_blocks: torch.Tensor):
        start_idx = len(self.on_site_real_block_pairs)
        blocks = on_site_blocks[:, start_idx:]
        return BlockConstructor.generate_block_sequence(blocks.unsqueeze(0), self.on_site_imag_block_pairs)
