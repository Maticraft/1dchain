import typing as t

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from src.models.base_models import MLP
from src.models.hamiltonian_handlers import BlockConstructor, BlockExtractor, get_matrix_from_strips, get_strip
from src.models.utils import diagonal_loss, edge_diff, eigenvectors_loss, kl_divergence_loss



class DistributionPreservingEncoder(nn.Module):
    def __init__(self, input_size: t.Tuple[int, int, int], representation_dim: t.Union[int, t.Tuple[int, int]], **kwargs: t.Dict[str, t.Any]):
        super(DistributionPreservingEncoder, self).__init__()
        self.channel_num = input_size[0]
        self.N = input_size[1]
        self.block_size = input_size[2]

        self.on_site_real_block_pairs = kwargs.get('on_site_real_block_pairs', ['z1', 'zx', 'zz', 'iyiy'])
        self.on_site_imag_block_pairs = kwargs.get('on_site_imag_block_pairs', ['1iy', 'xiy'])
        self.interaction_real_block_pairs = kwargs.get('interaction_real_block_pairs', ['z1', '1iy'])
        self.interaction_imag_block_pairs = kwargs.get('interaction_imag_block_pairs', ['z1', '1z', '1x']) # why for ladder z1 is used while for qdh it is 1z?

        self.total_on_site_params = len(self.on_site_real_block_pairs) + len(self.on_site_imag_block_pairs)
        self.num_independent_interaction_strips = (self.channel_num // 2 - 1) // 2
        self.interaction_params_per_strip = len(self.interaction_real_block_pairs) + len(self.interaction_imag_block_pairs)
        self.total_interaction_params = self.interaction_params_per_strip * self.num_independent_interaction_strips
        self.total_params = self.total_on_site_params + self.total_interaction_params
        self.total_distribution_params = 2*self.total_params # for mean and std
        self.encoded_data_dim = representation_dim - self.total_distribution_params
        
        self.seq_channels_num = kwargs.get('seq_channels_num', 64)
        self.enc_depth = kwargs.get('enc_depth', 4)
        self.enc_hidden_size = kwargs.get('enc_hidden_size', 128)

        self.seq_freq_enc_depth = kwargs.get('seq_freq_enc_depth', 2)
        self.seq_freq_hidden_size = kwargs.get('seq_freq_enc_hidden_size', 64)

        self.seq_enc_depth = kwargs.get('seq_enc_depth', 2)
        self.seq_enc_hidden_size = kwargs.get('seq_enc_hidden_size', 64)

        self.activation = kwargs.get('activation', 'relu')

        self.mixer = nn.Conv1d(self.total_params, self.seq_channels_num, kernel_size=1, stride=1)

        self.simple_parser = nn.ModuleList([
            MLP(self.seq_enc_depth, self.N, self.seq_enc_hidden_size, 1, self.activation)
            for _ in range(self.seq_channels_num)
        ])

        self.tf_seq_enc_layer = nn.TransformerEncoderLayer(d_model=self.seq_channels_num, nhead=2, dim_feedforward=128, batch_first=True)
        self.tf_seq_encoder = nn.TransformerEncoder(self.tf_seq_enc_layer, num_layers=1)
        self.seq_parser = nn.ModuleList([
            MLP(self.seq_enc_depth, self.N, self.seq_enc_hidden_size, 1, self.activation)
            for _ in range(self.seq_channels_num)
        ])

        self.fft_parser = nn.ModuleList([
            MLP(self.seq_freq_enc_depth, self.N, self.seq_freq_hidden_size, 1, self.activation)
            for _ in range(self.seq_channels_num)
        ])

        self.tf_freq_enc_layer = nn.TransformerEncoderLayer(d_model=self.seq_channels_num, nhead=2, dim_feedforward=128, batch_first=True)
        self.tf_freq_encoder = nn.TransformerEncoder(self.tf_freq_enc_layer, num_layers=1)
        self.freq_parser = nn.ModuleList([
            MLP(self.seq_freq_enc_depth, self.N, self.seq_freq_hidden_size, 1, self.activation)
            for _ in range(self.seq_channels_num)
        ])

        self.encoder = MLP(self.enc_depth, 4*self.seq_channels_num, self.enc_hidden_size, self.encoded_data_dim, self.activation)


    def forward(self, x: torch.Tensor):  
        seq = torch.cat(
            [self.extract_seq_from_matrix(x, i) for i in range(self.num_independent_interaction_strips + 1)],
            dim = 1
        ) # of shape (batch_size, total_params, N
        assert seq.shape[1] == self.total_params, f'Expected {self.total_params} parameters, got {seq.shape[1]}'

        seq, seq_mean, seq_std = self._normalize_seq(seq) # seq of shape (batch_size, total_params, N)
        distribution_data = torch.cat([seq_mean, seq_std], dim=-1) # of shape (batch_size, 2*total_params)
        
        masked_seq = self.mixer(seq) # of shape (batch_size, seq_channels_num, N)

        simple_seq_out = torch.cat([self.simple_parser[i](masked_seq[:, i, :]) for i in range(self.seq_channels_num)], dim=-1) # of shape (batch_size, seq_channels_num)

        block_strips = masked_seq.transpose(1, 2) # of shape (batch_size, N, seq_channels_num)
        tf_seq = self.tf_seq_encoder(block_strips) # of shape (batch_size, N, seq_channels_num)
        tf_seq_out = torch.cat([self.seq_parser[i](tf_seq[:, :, i]) for i in range(self.seq_channels_num)], dim=-1) # of shape (batch_size, seq_channels_num)

        complex_strips = torch.stack([torch.complex(masked_seq[:, i, :], masked_seq[:, self.seq_channels_num // 2 + i, :]) for i in range(self.seq_channels_num // 2)], dim=1) # of shape (batch_size, seq_channels_num // 2, N)
        fft_strips = torch.fft.fft(complex_strips, dim=-1) # of shape (batch_size, seq_channels_num // 2, N)
        fft_strips = torch.cat([fft_strips.real, fft_strips.imag], dim=1) # of shape (batch_size, seq_channels_num, N)

        fft_out = torch.cat([self.fft_parser[i](fft_strips[:, i, :]) for i in range(self.seq_channels_num)], dim=-1) # of shape (batch_size, seq_channels_num)

        fft_strips = fft_strips.transpose(1, 2) # of shape (batch_size, N, seq_channels_num)
        tf_freq_seq = self.tf_freq_encoder(fft_strips) # of shape (batch_size, N, seq_channels_num)
        tf_freq_out = torch.cat([self.freq_parser[i](tf_freq_seq[:, :, i]) for i in range(self.seq_channels_num)], dim=-1) # of shape (batch_size, seq_channels_num)

        encoded_data = self.encoder(torch.cat([simple_seq_out, tf_seq_out, fft_out, tf_freq_out], dim=-1)) # of shape (batch_size, encoded_data_dim)

        return torch.cat([distribution_data, encoded_data], dim=-1)

    def extract_seq_from_matrix(self, x: torch.Tensor, strip_idx: int):
        strip = get_strip(x, strip_idx, self.N, 'hamiltonian', self.block_size)
        if strip_idx == 0:
            seq_real = BlockExtractor.extract_block_sequences(strip[:, 0], self.on_site_real_block_pairs)
            seq_imag = BlockExtractor.extract_block_sequences(strip[:, 1], self.on_site_imag_block_pairs)
        else:
            seq_real = BlockExtractor.extract_block_sequences(strip[:, 0], self.interaction_real_block_pairs)
            seq_imag = BlockExtractor.extract_block_sequences(strip[:, 1], self.interaction_imag_block_pairs)
        seq = torch.cat([seq_real, seq_imag], dim=1)
        return seq
    
    def _normalize_seq(self, seq: torch.Tensor):
        '''
        seq.shape = (..., n_params, N)
        '''
        std_threshold = 1e-6
        seq_mean = seq.mean(dim=-1)
        # seq_std = torch.ones_like(seq_mean)
        seq_std = seq.var(dim=-1, unbiased=False) # actually it must be variance, because std causes nans (sqrt is not defined for 0 and smaller values)
        seq_std = torch.maximum(seq_std, torch.full_like(seq_std, 1.e-6))
        normalized_seq = (seq - seq_mean.unsqueeze(-1)) / seq_std.unsqueeze(-1)
        # normalized_seq_masked = torch.where(seq_std.unsqueeze(-1) < std_threshold, torch.zeros_like(normalized_seq), normalized_seq)
        return normalized_seq, seq_mean, seq_std


class DistributionPreservingHamiltonianGenerator(nn.Module):
    def __init__(self, representation_dim: int, output_size: t.Tuple[int, int, int], **kwargs: t.Dict[str, t.Any]):
        super(DistributionPreservingHamiltonianGenerator, self).__init__()
        self.channel_num = output_size[0]
        assert self.channel_num % 2 == 0, 'Channel number must be even'
        self.N = output_size[1]
        self.block_size = output_size[2]

        self.on_site_real_block_pairs = kwargs.get('on_site_real_block_pairs', ['z1', 'zx', 'zz', 'iyiy'])
        self.on_site_imag_block_pairs = kwargs.get('on_site_imag_block_pairs', ['1iy', 'xiy'])
        self.interaction_real_block_pairs = kwargs.get('interaction_real_block_pairs', ['z1', '1iy'])
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
            nn.Conv1d(self.seq_channels_num, self.interaction_params_per_strip, kernel_size=2, stride=1, dilation=interaction_range, bias=False)
            for interaction_range in range(1, self.num_independent_interaction_strips + 1)
        ])
        self.allow_periodic = kwargs.get('allow_periodic', False)
        self.interlevels_interactions = kwargs.get('interlevels_interactions', False)
        self.smoothing = kwargs.get('smoothing', False)


    def forward(self, x: torch.Tensor):
        '''
        correct order:
            x = [distribution_data, encoded_data]
            distribution_data = [
                distribution_mean [
                    on_site [real, imag],
                    interactions [upper strips from 0 to num_independent_interaction_strips-1 [real, imag]]
                ],
                distribution_std [...]
            ]
        '''
        distribution_data = x[:, :self.total_distribution_params]
        encoded_data = x[:, self.total_distribution_params:]

        on_site_seq, interactions_seq = self._propagate_neural_network(encoded_data)
        
        distribution_mean = distribution_data[:, :self.total_params]
        distribution_std = distribution_data[:, self.total_params:]
        on_site_mean = distribution_mean[:, :self.total_on_site_params]
        on_site_std = distribution_std[:, :self.total_on_site_params]
        interactions_mean = distribution_mean[:, self.total_on_site_params:].view(-1, self.num_independent_interaction_strips, self.interaction_params_per_strip)
        interactions_std = distribution_std[:, self.total_on_site_params:].view(-1, self.num_independent_interaction_strips, self.interaction_params_per_strip)

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
        right_upper_coeff = lower_strip[:, :, 0, 1::4].clone()
        left_upper_coeff = lower_strip[:, :, 1, 0::4].clone()
        lower_strip[:, :, 0, 1::4] = left_upper_coeff
        lower_strip[:, :, 1, 0::4] = right_upper_coeff

        right_lower_coeff = lower_strip[:, :, 2, 3::4].clone()
        left_lower_coeff = lower_strip[:, :, 3, 2::4].clone()
        lower_strip[:, :, 2, 3::4] = left_lower_coeff
        lower_strip[:, :, 3, 2::4] = right_lower_coeff
        return lower_strip
    
    def _interaction_real_block_generator(self, interations_blocks: torch.Tensor):
        '''
        assumes interations_blocks.shape = (batch_size, interaction_params, seq_size)
        '''
        end_indx = len(self.interaction_real_block_pairs)
        blocks = interations_blocks[:, :end_indx]
        return BlockConstructor.generate_block_sequence(blocks.transpose(0, 1), self.interaction_real_block_pairs)

    def _interaction_imag_block_generator(self, interations_blocks: torch.Tensor):
        '''
        assumes interations_blocks.shape = (batch_size, interaction_params, seq_size)
        '''
        start_indx = len(self.interaction_real_block_pairs)
        blocks = interations_blocks[:, start_indx:]
        return BlockConstructor.generate_block_sequence(blocks.transpose(0, 1), self.interaction_imag_block_pairs)


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
        return BlockConstructor.generate_block_sequence(blocks.transpose(0, 1), self.on_site_imag_block_pairs)



class VariationalDistributionPreservingEncoder(DistributionPreservingEncoder):
    def __init__(self, input_size: t.Tuple[int, int, int], representation_dim: t.Union[int, t.Tuple[int, int]], **kwargs: t.Dict[str, t.Any]):
        new_representation_dim = 2*representation_dim if isinstance(representation_dim, int) else (2*representation_dim[0], 2*representation_dim[1])
        super(VariationalDistributionPreservingEncoder, self).__init__(input_size, new_representation_dim, **kwargs)

    def forward(self, x: torch.Tensor, return_distr: bool = False, eps: float = 1e-10):
        x = super(VariationalDistributionPreservingEncoder, self).forward(x)
        x_mean, x_std = torch.split(x, x.shape[-1] // 2, dim=-1)
        x_dist = Normal(x_mean, x_std.exp() + eps)
        x_sample = x_dist.rsample()
        if return_distr:
            return x_sample, x_dist
        else:
            return x_sample



def train_vae(
    vencoder_model: nn.Module,
    decoder_model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    epoch: int,
    device: torch.device,
    vencoder_optimizer: torch.optim.Optimizer,
    decoder_optimizer: torch.optim.Optimizer,
    edge_loss: bool = False,
    edge_loss_weight: float = .5,
    eigenstates_loss: bool = False,
    eigenstates_loss_weight: float = .5,
    diag_loss: bool = False,
    diag_loss_weight: float = .01,
    kl_loss: bool = True,
    kl_loss_weight: float = 0.01,
):
    criterion = nn.MSELoss()

    vencoder_model.to(device)
    decoder_model.to(device)

    vencoder_model.train()
    decoder_model.train()

    total_reconstruction_loss = 0
    total_edge_loss = 0
    total_eigenstates_loss = 0
    total_diag_loss = 0
    total_kl_loss = 0

    print(f'Epoch: {epoch}')
    for (x, _), eig_dec in tqdm(train_loader, 'Training model'):
        x = x.to(device)
        vencoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        z, z_dist = vencoder_model(x, return_distr = True)
        x_hat = decoder_model(z)
        loss = criterion(x_hat, x)
        total_reconstruction_loss += torch.mean(loss).item()

        if kl_loss:
            k1_loss = kl_divergence_loss(z_dist).mean()
            total_kl_loss += k1_loss.item()
            loss += kl_loss_weight * k1_loss

        if edge_loss:
            e_loss = edge_diff(x_hat, x, criterion, edge_width=8)
            loss += edge_loss_weight * e_loss
            total_edge_loss += torch.mean(e_loss).item()

        if eigenstates_loss:
            assert eig_dec is not None, "Incorrect eigen decomposition values"
            eig_dec = eig_dec[0].to(device), eig_dec[1].to(device)
            eig_loss = eigenvectors_loss(x_hat, eig_dec, criterion)
            loss += eigenstates_loss_weight * eig_loss
            total_eigenstates_loss += torch.mean(eig_loss).item()

        if diag_loss:
            diag_loss = diagonal_loss(x_hat, x, criterion, block_size=4)
            loss += diag_loss_weight * diag_loss
            total_diag_loss += torch.mean(diag_loss).item()

        loss.backward()
        vencoder_optimizer.step()
        decoder_optimizer.step()

    total_reconstruction_loss /= len(train_loader)
    total_edge_loss /= len(train_loader)
    total_eigenstates_loss /= len(train_loader)
    total_diag_loss /= len(train_loader)
    total_kl_loss /= len(train_loader)

    print(f'Reconstruction Loss: {total_reconstruction_loss}')
    print(f'KL Loss: {total_kl_loss}')
    if edge_loss:
        print(f'Edge Loss: {total_edge_loss}')
    if eigenstates_loss:
        print(f'Eigenstates Loss: {total_eigenstates_loss}')
    if diag_loss:
        print(f'Diagonal Loss: {total_diag_loss}')
    print()

    return total_reconstruction_loss, total_kl_loss, total_edge_loss, total_eigenstates_loss, total_diag_loss