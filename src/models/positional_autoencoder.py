import torch
import torch.nn as nn


import typing as t


class PositionalDecoder(nn.Module):
    def __init__(self, representation_dim: t.Union[int, t.Tuple[int, int]], output_size: t.Tuple[int, int, int], **kwargs: t.Dict[str, t.Any]):
        super(PositionalDecoder, self).__init__()
        self.channel_num = output_size[0]
        self.N = output_size[1]
        self.block_size = output_size[2]
        if type(representation_dim) == int:
            self.freq_dim = representation_dim // 2
            self.block_dim = representation_dim // 2
        else:
            self.freq_dim = representation_dim[0]
            self.block_dim = representation_dim[1]

        self.kernel_size = self.block_size
        self.stride = self.block_size
        self.dilation = 1
        self.site_size = (1, self.N)

        self.kernel_num = kwargs.get('kernel_num', 32)

        self.freq_dec_depth = kwargs.get('freq_dec_depth', 4)
        self.freq_dec_hidden_size = kwargs.get('freq_dec_hidden_size', 128)

        self.block_dec_depth = kwargs.get('block_dec_depth', 4)
        self.block_dec_hidden_size = kwargs.get('block_dec_hidden_size', 128)

        self.seq_dec_depth = kwargs.get('seq_dec_depth', 4)
        self.seq_dec_hidden_size = kwargs.get('seq_dec_hidden_size', 128)

        self.activation = kwargs.get('activation', 'relu')

        self.strip_len = self._get_convs_input_size(1)

        self.freq_decoder = self._get_mlp(self.freq_dec_depth, self.freq_dim, self.freq_dec_hidden_size, self.freq_dec_hidden_size)
        self.freq_seq_constructor = nn.ModuleList([
            self._get_mlp(self.freq_dec_depth, self.freq_dec_hidden_size, self.freq_dec_hidden_size, self.strip_len, final_activation='sigmoid')
            for _ in range(self.kernel_num)
        ])

        self.naive_block_decoder = self._get_mlp(self.block_dec_depth, self.block_dim, self.block_dec_hidden_size, self.kernel_num)

        self.naive_seq_decoder = self._get_mlp(self.seq_dec_depth, self.freq_dim+self.block_dim, self.seq_dec_hidden_size, self.strip_len)

        self.tf_seq_decoder_layer = nn.TransformerEncoderLayer(d_model=self.kernel_num, nhead=2, dim_feedforward=128, batch_first=True)
        self.tf_seq_decoder = nn.TransformerEncoder(self.tf_seq_decoder_layer, num_layers=1)

        self.conv = self._get_conv_block()


    def _get_activation(self):
        if self.activation == 'relu':
            return nn.ReLU()
        elif self.activation == 'leaky_relu':
            return nn.LeakyReLU(negative_slope=0.01)
        else:
            return ValueError(f'Activation function: {self.activation} not implemented')


    def _get_conv_block(self):
        return nn.Sequential(
            nn.ConvTranspose2d(2*self.kernel_num+1, self.channel_num, kernel_size=self.kernel_size, stride=self.stride, dilation=self.dilation),
        )


    def _get_convs_input_size(self, dim):
        return (self.site_size[dim]*self.block_size - self.dilation*(self.kernel_size-1) - 1) // self.stride + 1


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

            if i != layers_num - 1:
                layers.append(self._get_activation())

        if final_activation == 'self':
            layers.append(self._get_activation())
        elif final_activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        elif final_activation == 'tanh':
            layers.append(nn.Tanh())
        else:
            raise ValueError(f'Activation function: {final_activation} not implemented')

        return nn.Sequential(*layers)


    def _periodic_func(self, x: torch.Tensor, i: int):
        if i % 2 == 0:
            return torch.sin(x * 2**((i // 2) / 4))
        else:
            return torch.cos(x * 2**((i // 2) / 4))


    def forward(self, x: torch.Tensor):
        block = self.naive_block_decoder(x[:, self.freq_dim:]).unsqueeze(0)
        block_expand = block.expand(self.strip_len, -1, -1).permute((1, 2, 0)).unsqueeze(2)

        freq = self.freq_decoder(x[:, :self.freq_dim])
        freq_seq = torch.stack([self._periodic_func(self.freq_seq_constructor[i](freq), i) for i in range(self.kernel_num)], dim=-1)

        tf_input = freq_seq * block_expand.squeeze(2).transpose(1, 2)
        seq = self.tf_seq_decoder(tf_input)
        seq = seq.transpose(1, 2).unsqueeze(2)

        naive_seq = self.naive_seq_decoder(x).unsqueeze(1).unsqueeze(1)

        block_seq = torch.cat([block_expand, seq, naive_seq], dim=1)

        strips = self.conv(block_seq)
        matrix = self._get_matrix_from_strips(strips)
        return matrix


class PositionalEncoder(nn.Module):
    def __init__(self, input_size: t.Tuple[int, int, int], representation_dim: t.Union[int, t.Tuple[int, int]], **kwargs: t.Dict[str, t.Any]):
        super(PositionalEncoder, self).__init__()
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
            self._get_mlp(self.block_enc_depth, self.strip_len, self.block_enc_hidden_size, 1)
            for _ in range(self.kernel_num)
        ])
        self.simple_encoder = nn.Sequential(
            nn.Linear(self.kernel_num, self.block_dim),
            self._get_activation(),
        )

        self.tf_block_enc_layer = nn.TransformerEncoderLayer(d_model=self.kernel_num, nhead=2, dim_feedforward=128, batch_first=True)
        self.tf_block_encoder = nn.TransformerEncoder(self.tf_block_enc_layer, num_layers=1)
        self.block_parser = nn.ModuleList([
            self._get_mlp(self.block_enc_depth, self.strip_len, self.block_enc_hidden_size, 1)
            for _ in range(self.kernel_num)
        ])
        self.block_encoder = nn.Sequential(
            nn.Linear(self.kernel_num, self.block_dim),
            self._get_activation(),
        )

        self.fft_parser = nn.ModuleList([
            self._get_mlp(self.freq_enc_depth, self.strip_len, self.freq_enc_hidden_size, 1)
            for _ in range(self.kernel_num)
        ])
        self.fft_encoder = nn.Sequential(
            nn.Linear(self.kernel_num, self.freq_dim),
            self._get_activation(),
        )

        self.tf_freq_enc_layer = nn.TransformerEncoderLayer(d_model=self.kernel_num, nhead=2, dim_feedforward=128, batch_first=True)
        self.tf_freq_encoder = nn.TransformerEncoder(self.tf_freq_enc_layer, num_layers=1)
        self.freq_parser = nn.ModuleList([
            self._get_mlp(self.freq_enc_depth, self.strip_len, self.freq_enc_hidden_size, 1)
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
            self._get_activation(),
            nn.BatchNorm2d(self.kernel_num),
        )


    def _get_convs_output_size(self, dim):
        return (self.site_size[dim]*self.block_size - self.dilation*(self.block_size-1) - 1) // self.stride + 1

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
            layers.append(self._get_activation())
        return nn.Sequential(*layers)


    def _get_strip(self, x: torch.Tensor, offset: int, fill_mode: str = 'zeros'):
        strip = torch.zeros((x.shape[0], x.shape[1], self.block_size, self.N*self.block_size)).to(x.device)
        x_off = abs(offset)*self.block_size
        for i in range(self.N):
            idx0 =  i*self.block_size
            idx1 = idx0 + self.block_size
            if offset >= 0:
                idx0_off = (idx0 + x_off) % (self.N * self.block_size)
                idx1_off = idx0_off + self.block_size
            else:
                idx0_off = (idx0 - x_off) % (self.N * self.block_size)
                idx1_off = idx0_off + self.block_size
            strip[:, :, :, idx0: idx1] = x[:, :, idx0: idx1, idx0_off: idx1_off]
        if fill_mode == 'zeros':
            if offset > 0:
                strip[:, :, :, -x_off:] = 0.
            elif offset < 0:
                strip[:, :, :, :x_off] = 0.
            return strip
        elif fill_mode == 'circular':
            if offset > 0:
                strip[:, :, :, -x_off:] = strip[:, :, :, :x_off]
            elif offset < 0:
                strip[:, :, :, :x_off] = strip[:, :, :, -x_off:]
            return strip
        elif fill_mode == 'hamiltonian':
            return strip
        else:
            raise ValueError(f'Fill mode: {fill_mode} not implemented')


    def forward(self, x: torch.Tensor):
        strip_bound = ((self.channel_num // 2) - 1) // 2
        x = torch.cat([self._get_strip(x, i, 'hamiltonian') for i in range(-strip_bound, strip_bound + 1)], dim=1)
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