import typing as t
from itertools import product

import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from models_utils import get_edges

class Classifier(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 1, layers: int = 2, hidden_size: int = 128):
        super(Classifier, self).__init__()
        self.mlp = self._get_mlp(layers, input_dim, hidden_size, output_dim)
        self.sigmoid = nn.Sigmoid()

    def _get_mlp(self, layers: int, input_dim: int, hidden_size: int, output_dim: int):
        mlp = []
        mlp.append(nn.Linear(input_dim, hidden_size))
        mlp.append(nn.ReLU())
        for i in range(layers - 1):
            mlp.append(nn.Linear(hidden_size, hidden_size))
            mlp.append(nn.ReLU())
        mlp.append(nn.Linear(hidden_size, output_dim))
        return nn.Sequential(*mlp)

    def forward(self, x: torch.Tensor):
        return self.sigmoid(self.mlp(x))


class Decoder(nn.Module):
    def __init__(self, representation_dim: int, output_size: t.Tuple[int, int, int],  **kwargs: t.Dict[str, t.Any]):
        '''
        Parameters:
        - input_size: (channels_num, N, block_size)
        - representation_dim: dimension of output representation
        - kwargs:
            - fc_num: number of fully connected layers
            - upsample_method: method of upsampling (possible values: 'transpose', 'nearest', 'bilinear')
            - conv_num: number of upsample convolutional layers
            - kernel_size: kernel size of convolutional layer
            - stride: stride of convolutional layer
            - dilation: dilation of convolutional layer
            - kernel_num: number of kernels
            - hidden_size: number of hidden units
            - use_strips: flag of using either full matrix or non-zero diagonal strips only

        '''

        super(Decoder, self).__init__()
        self.channel_num = output_size[0]
        self.N = output_size[1]
        self.block_size = output_size[2]
        self.representation_dim = representation_dim

        self.fc_num = kwargs.get('fc_num', 2)
        self.hidden_size = kwargs.get('hidden_size', 128)

        self.upsample_method = kwargs.get('upsample_method', 'transpose')
        
        self.conv_num = kwargs.get('conv_num', 1)
        self.kernel_size = self._format_2d_size(kwargs.get('kernel_size', self.block_size))
        self.kernel_size1 = self._format_2d_size(kwargs.get('kernel_size1', self.kernel_size))
        self.stride = self._format_2d_size(kwargs.get('stride', self.block_size))
        self.stride1 = self._format_2d_size(kwargs.get('stride1', self.stride))
        self.dilation = self._format_2d_size(kwargs.get('dilation', 1))
        self.dilation1 = self._format_2d_size(kwargs.get('dilation1', self.dilation))
        self.kernel_num = kwargs.get('kernel_num', 32)
        self.kernel_num1 = kwargs.get('kernel_num1', self.kernel_num)
        self.activation = kwargs.get('activation', 'relu')
        self.use_strips = kwargs.get('use_strips', False)

        if self.use_strips:
            self.site_size = (1, self.N)
        else:
            self.site_size = (self.N, self.N)

        sf = kwargs.get('scale_factor', 2*self.stride)
        self.scale_factor = [self._format_2d_size(x) for x in self._get_scale_factor(sf)]

        self.convs_input_size = int(self._get_convs_input_size(0)), int(self._get_convs_input_size(1))

        if self.conv_num > 1:
            self.fcs_output_size = (self.convs_input_size[0] * self.convs_input_size[1]) * self.kernel_num
        else:
            self.fcs_output_size = (self.convs_input_size[0] * self.convs_input_size[1]) * self.kernel_num1

        self.fcs = self._get_fcs()
        self.convs = self._get_convs()

    def _format_2d_size(self, x: t.Any):
        if type(x) == tuple:
            return x
        elif type(x) == list:
            return tuple(x)
        elif type(x) == int:
            return (x, x)
        else:
            raise TypeError("Wrong type of kernel size")
        
    def _get_activation(self):
        if self.activation == 'relu':
            return nn.ReLU()
        elif self.activation == 'leaky_relu':
            return nn.LeakyReLU(negative_slope=0.01)
        else:
            return ValueError(f'Activation function: {self.activation} not implemented')

    def _get_convs(self):
        convs = []
        if self.upsample_method == 'transpose':
            for _ in range(2, self.conv_num):
                convs.append(nn.ConvTranspose2d(self.kernel_num, self.kernel_num, kernel_size=self.kernel_size, stride=self.stride, dilation=self.dilation))
                convs.append(self._get_activation())
                convs.append(nn.BatchNorm2d(self.kernel_num))
            if self.conv_num > 1:
                convs.append(nn.ConvTranspose2d(self.kernel_num, self.kernel_num1, kernel_size=self.kernel_size, stride=self.stride, dilation=self.dilation))
                convs.append(self._get_activation())
                convs.append(nn.BatchNorm2d(self.kernel_num1))

            convs.append(nn.ConvTranspose2d(self.kernel_num1, self.channel_num, kernel_size=self.kernel_size1, stride=self.stride1, dilation=self.dilation1))
        else:
            for i in range(1, self.conv_num - 1):
                if self.stride > 1:
                    raise ValueError("Upsample not implemented for stride larger than 1")
                convs.append(nn.Upsample(scale_factor=self.scale_factor[-i], mode=self.upsample_method))
                convs.append(nn.Conv2d(self.kernel_num, self.kernel_num, kernel_size=self.kernel_size, stride=self.stride, dilation=self.dilation, padding='same'))
                convs.append(self._get_activation())
                convs.append(nn.BatchNorm2d(self.kernel_num))
            if self.conv_num > 1:
                convs.append(nn.Upsample(scale_factor=self.scale_factor[1], mode=self.upsample_method))
                convs.append(nn.Conv2d(self.kernel_num, self.kernel_num1, kernel_size=self.kernel_size, stride=self.stride, dilation=self.dilation, padding='same'))
                convs.append(self._get_activation())
                convs.append(nn.BatchNorm2d(self.kernel_num1))

            convs.append(nn.Upsample(scale_factor=self.scale_factor[0], mode=self.upsample_method))
            convs.append(nn.Conv2d(self.kernel_num1, self.channel_num, kernel_size=self.kernel_size1, stride=self.stride1, dilation=self.dilation1, padding='same'))

        return nn.Sequential(*convs)

    def _get_convs_input_size(self, dim):
        if self.upsample_method == 'transpose':
            input_size = (self.site_size[dim]*self.block_size - self.dilation1[dim]*(self.kernel_size1[dim]-1) - 1) // self.stride1[dim] + 1
            for _ in range(1, self.conv_num):
                input_size = (input_size - self.dilation[dim]*(self.kernel_size[dim]-1) - 1) // self.stride[dim] + 1
        else:
            input_size = round(self.site_size[dim]*self.block_size / self.scale_factor[0][dim])
            for i in range(1, self.conv_num):
                input_size = round(input_size / self.scale_factor[i][dim])
        return input_size

    def _get_fcs(self):
        fcs = []
        fcs.append(nn.Linear(self.representation_dim, self.hidden_size))
        fcs.append(self._get_activation())
        fcs.append(nn.BatchNorm1d(self.hidden_size))
        for _ in range(1, self.fc_num - 1):
            fcs.append(nn.Linear(self.hidden_size, self.hidden_size))
            fcs.append(self._get_activation())
            fcs.append(nn.BatchNorm1d(self.hidden_size))
        fcs.append(nn.Linear(self.hidden_size, self.fcs_output_size))
        return nn.Sequential(*fcs)

    def _get_scale_factor(self, sf: t.Any):
        if type(sf) == float or type(sf) == int or type(sf) == tuple:
            return [sf for _ in range(self.conv_num)]
        elif type(sf) == list:
            if len(sf) != self.conv_num:
                raise ValueError("Wrong number of scale factors")
            return sf
        else:
            raise TypeError("Wrong type of scale factor: must be either int, float or list")
        
    def _get_matrix_from_strips(self, strips: torch.Tensor):
        matrix = torch.zeros((strips.shape[0], 2, self.N*self.block_size, self.N*self.block_size)).to(strips.device)
        strips_split = torch.tensor_split(strips, self.channel_num // 2, dim=1)
        for i, strip in enumerate(strips_split):
            offset = i - (len(strips_split) // 2)
            strip_off = max(0, -offset)
            matrix_off = abs(offset)*self.block_size
            for j in range(self.N - abs(offset)):
                idx0 =  j*self.block_size
                idx1 = (j+1)*self.block_size
                if offset >= 0:
                    matrix[:, :, idx0: idx1, idx0 + matrix_off: idx1 + matrix_off] = strip[:, :, :, idx0 + strip_off: idx1 + strip_off]
                else:
                    matrix[:, :, idx0 + matrix_off: idx1 + matrix_off, idx0: idx1] = strip[:, :, :, idx0 + strip_off: idx1 + strip_off]
        return matrix

    def forward(self, x: torch.Tensor):
        x = self.fcs(x)
        x = x.view(-1, self.kernel_num, self.convs_input_size[0], self.convs_input_size[1])
        x = self.convs(x)
        if self.use_strips:
            x = self._get_matrix_from_strips(x)  
        return x


class DecoderEnsemble(nn.Module):
    def __init__(self,
    representation_dim: int,
    output_size: t.Tuple[int, int, int],
    decoders_params: t.Dict[str, t.Any],
):
        super(DecoderEnsemble, self).__init__()

        decoders_num = decoders_params.get('decoders_num', 1)
        edge_decoder_idx = decoders_params.get('edge_decoder_idx', None)

        if not edge_decoder_idx:
            self.edge_decoder_ids = []
        elif type(edge_decoder_idx) == int:
            self.edge_decoder_ids = [edge_decoder_idx]
        else:
            self.edge_decoder_ids = edge_decoder_idx

        self.decoders = nn.ModuleList([Decoder(representation_dim, output_size, **decoders_params[f'decoder_{i}']) for i in range(decoders_num) if i not in self.edge_decoder_ids])
        if self.edge_decoder_ids:
            self.edge_decoders = nn.ModuleList([Decoder(representation_dim, output_size, **decoders_params[f'decoder_{i}']) for i in self.edge_decoder_ids])
        
        self.ensembler = nn.Conv2d(decoders_num*output_size[0], output_size[0], kernel_size=1)

    def forward(self, x: torch.Tensor):
        if self.edge_decoder_ids:
            ezs = [get_edges(decoder(x), edge_width=8) for decoder in self.edge_decoders]
        else:
            ezs = []
            
        zs = [decoder(x) for decoder in self.decoders] + ezs

        z = torch.cat(zs, dim=1)
        return self.ensembler(z)


class Encoder(nn.Module):
    def __init__(self, input_size: t.Tuple[int, int, int], representation_dim: int, **kwargs: t.Dict[str, t.Any]):
        '''
        Parameters:
        - input_size: (channels_num, N, block_size)
        - representation_dim: dimension of output representation
        - kwargs:
            - fc_num: number of fully connected layers
            - conv_num: number of convolutional layers
            - kernel_size: kernel size of convolutional layer
            - stride: stride of convolutional layer
            - dilation: dilation of convolutional layer
            - kernel_num: number of kernels
            - hidden_size: number of hidden units
            - use_strips: flag of using either full matrix or non-zero diagonal strips only
        '''
        super(Encoder, self).__init__()
        self.channel_num = input_size[0]
        self.N = input_size[1]
        self.block_size = input_size[2]
        self.representation_dim = representation_dim

        self.kernel_size = self._format_2d_size(kwargs.get('kernel_size', self.block_size))
        self.kernel_size1 = self._format_2d_size(kwargs.get('kernel_size1', self.kernel_size))
        self.stride = self._format_2d_size(kwargs.get('stride', self.block_size))
        self.stride1 = self._format_2d_size(kwargs.get('stride1', self.stride))
        self.dilation = self._format_2d_size(kwargs.get('dilation', 1))
        self.dilation1 = self._format_2d_size(kwargs.get('dilation1', self.dilation))
        self.fc_num = kwargs.get('fc_num', 2)
        self.conv_num = kwargs.get('conv_num', 1)
        self.kernel_num = kwargs.get('kernel_num', 32)
        self.kernel_num1 = kwargs.get('kernel_num1', self.kernel_num)
        self.hidden_size = kwargs.get('hidden_size', 128)
        self.activation = kwargs.get('activation', 'relu')
        self.use_strips = kwargs.get('use_strips', False)

        if self.use_strips:
            self.site_size = (1, self.N)
        else:
            self.site_size = (self.N, self.N)

        if self.conv_num > 1:
            self.convs_output_size = (self._get_convs_output_size(0) * self._get_convs_output_size(1)) * self.kernel_num
        else:
            self.convs_output_size = (self._get_convs_output_size(0) * self._get_convs_output_size(1)) * self.kernel_num1

        self.convs = self._get_convs()
        self.fcs = self._get_fcs()

    def _format_2d_size(self, x: t.Any):
        if type(x) == tuple:
            return x
        elif type(x) == list:
            return tuple(x)
        elif type(x) == int:
            return (x, x)
        else:
            raise TypeError("Wrong type of kernel size")

    def _get_activation(self):
        if self.activation == 'relu':
            return nn.ReLU()
        elif self.activation == 'leaky_relu':
            return nn.LeakyReLU(negative_slope=0.01)
        else:
            return ValueError(f'Activation function: {self.activation} not implemented')

    def _get_convs(self):
        convs = []
        convs.append(nn.Conv2d(self.channel_num, self.kernel_num1, kernel_size= self.kernel_size1, stride=self.stride1, dilation=self.dilation1))
        convs.append(self._get_activation())
        convs.append(nn.BatchNorm2d(self.kernel_num1))

        if self.conv_num > 1:
            convs.append(nn.Conv2d(self.kernel_num1, self.kernel_num, kernel_size= self.kernel_size, stride=self.stride, dilation=self.dilation))
            convs.append(self._get_activation())
            convs.append(nn.BatchNorm2d(self.kernel_num))

        for _ in range(2, self.conv_num):
            convs.append(nn.Conv2d(self.kernel_num, self.kernel_num, kernel_size=self.kernel_size, stride=self.stride, dilation=self.dilation))
            convs.append(self._get_activation())
            convs.append(nn.BatchNorm2d(self.kernel_num))
        return nn.Sequential(*convs)

    def _get_convs_output_size(self, dim):
        output_size = (self.site_size[dim]*self.block_size - self.dilation1[dim]*(self.kernel_size1[dim]-1) - 1) // self.stride1[dim] + 1
        for _ in range(1, self.conv_num):
            output_size = (output_size - self.dilation[dim]*(self.kernel_size[dim]-1) - 1) // self.stride[dim] + 1
        return output_size
    
    def _get_fcs(self):
        fcs = []
        fcs.append(nn.Linear(self.convs_output_size, self.hidden_size))
        fcs.append(self._get_activation())
        fcs.append(nn.BatchNorm1d(self.hidden_size))
        for _ in range(1, self.fc_num - 1):
            fcs.append(nn.Linear(self.hidden_size, self.hidden_size))
            fcs.append(self._get_activation())
            fcs.append(nn.BatchNorm1d(self.hidden_size))
        fcs.append(nn.Linear(self.hidden_size, self.representation_dim))
        return nn.Sequential(*fcs)

    def _get_strip(self, x: torch.Tensor, offset: int):
        strip = torch.zeros((x.shape[0], x.shape[1], self.block_size, self.N*self.block_size)).to(x.device)
        strip_off = max(0, -offset)
        idx_off = abs(offset)*self.block_size
        for i in range(self.N - abs(offset)):
            idx0 =  i*self.block_size
            idx1 = (i+1)*self.block_size
            if offset >= 0:
                strip[:, :, :, idx0 + strip_off: idx1 + strip_off] = x[:, :, idx0: idx1, idx0 + idx_off: idx1 + idx_off]
            else:
                strip[:, :, :, idx0 + strip_off: idx1 + strip_off] = x[:, :, idx0 + idx_off: idx1 + idx_off, idx0: idx1]
        return strip
    
    def forward(self, x: torch.Tensor):
        if self.use_strips:
            strip_bound = ((self.channel_num // 2) - 1) // 2
            x = torch.cat([self._get_strip(x, i) for i in range(-strip_bound, strip_bound + 1)], dim=1)
        x = self.convs(x)
        x = x.view(-1, self.convs_output_size)
        x = self.fcs(x)
        return x


class EncoderEnsemble(nn.Module):
    def __init__(self,
        input_size: t.Tuple[int, int, int],
        representation_dim: int,
        encoders_params: t.Dict[str, t.Any],
    ):
        super(EncoderEnsemble, self).__init__()

        encoders_num = encoders_params.get('encoders_num', 1)
        edge_encoder_idx = encoders_params.get('edge_encoder_idx', None)

        if not edge_encoder_idx:
            self.edge_encoder_ids = []
        elif type(edge_encoder_idx) == int:
            self.edge_encoder_ids = [edge_encoder_idx]
        else:
            self.edge_encoder_ids = edge_encoder_idx

        self.encoders = nn.ModuleList([Encoder(input_size, representation_dim, **encoders_params[f'encoder_{i}']) for i in range(encoders_num) if i not in self.edge_encoder_ids])
        if self.edge_encoder_ids:
            self.edge_encoders = nn.ModuleList([Encoder(input_size, representation_dim, **encoders_params[f'encoder_{i}']) for i in self.edge_encoder_ids])
        self.ensembler = nn.Linear(encoders_num*representation_dim, representation_dim)

    def forward(self, x: torch.Tensor):
        if self.edge_encoder_ids:
            edges = get_edges(x, edge_width=8)
            ezs = [encoder(edges) for encoder in self.edge_encoders]
        else:
            ezs = []
        zs = [encoder(x) for encoder in self.encoders] + ezs
        z = torch.cat(zs, dim=-1)
        return self.ensembler(z)


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
    


class VariationalPositionalEncoder(PositionalEncoder):
    def __init__(self, input_size: t.Tuple[int, int, int], representation_dim: t.Union[int, t.Tuple[int, int]], **kwargs: t.Dict[str, t.Any]):
        new_representation_dim = 2*representation_dim if isinstance(representation_dim, int) else (2*representation_dim[0], 2*representation_dim[1])
        super(VariationalPositionalEncoder, self).__init__(input_size, new_representation_dim, **kwargs)

    def forward(self, x: torch.Tensor, return_distr: bool = False, eps: float = 1e-10):
        x = super(VariationalPositionalEncoder, self).forward(x)
        freq_enc, block_enc = x[..., :self.freq_dim], x[..., self.freq_dim:]
        freq_mu, freq_std = torch.split(freq_enc, self.freq_dim // 2, dim=-1)
        block_mu, block_std = torch.split(block_enc, self.block_dim // 2, dim=-1)
        freq_dist = Normal(freq_mu, freq_std.exp() + eps)
        block_dist = Normal(block_mu, block_std.exp() + eps)
        freq_sample = freq_dist.rsample()
        block_sample = block_dist.rsample()
        sample = torch.cat([freq_sample, block_sample], dim=-1)
        if return_distr:
            return sample, (freq_dist, block_dist)
        else:
            return sample
        

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


class Discriminator(nn.Module):
    def __init__(self, model_class: t.Type[nn.Module], model_config: t.Dict[str, t.Any]):
        super(Discriminator, self).__init__()
        self.nn = model_class(**model_config)
        self.nn_out_features = model_config['representation_dim'] if isinstance(model_config['representation_dim'], int) else int(sum(model_config['representation_dim']))
        self.classifier = self._get_mlp(3, self.nn_out_features, 128, 1)
    
    def forward(self, x: torch.Tensor):
        out = self.nn(x)
        return self.classifier(out)
    
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


class Generator(nn.Module):
    def __init__(self, model_class: t.Type[nn.Module], model_config: t.Dict[str, t.Any], **kwargs: t.Dict[str, t.Any]):
        super(Generator, self).__init__()
        self.nn_in_features = model_config['representation_dim'] if isinstance(model_config['representation_dim'], int) else int(sum(model_config['representation_dim']))
        self.nn = model_class(**model_config)
        if 'distribution' in kwargs:
            activation = self._create_activation_from_distribution(kwargs['distribution'])
        else:
            activation = nn.LeakyReLU(negative_slope=0.01)

        self.noise_converter = self._get_mlp(5, self.nn_in_features, self.nn_in_features, self.nn_in_features, final_activation=activation)
    
    def forward(self, x: torch.Tensor):
        x = self.noise_converter(x)
        return self.nn(x)
    
    def _create_activation_from_distribution(self, distribution: t.Tuple[torch.Tensor, torch.Tensor]):
        mu, std = distribution
        class Activation(nn.Module):
            def __init__(self, mu: torch.Tensor, std: torch.Tensor):
                super(Activation, self).__init__()
                self.mu = mu
                self.std = std
            def activation(self, x: torch.Tensor):
                mu = self.mu.to(x.device)
                std = self.std.to(x.device)
                min_cutoff = torch.maximum(x, mu - 2*std)
                max_cutoff = torch.minimum(min_cutoff, mu + 2*std)
                return max_cutoff
            def forward(self, x: torch.Tensor):
                return self.activation(x)
        return Activation(mu, std)
    
    def _get_mlp(self, layers_num: int, input_size: int, hidden_size: int, output_size: int, final_activation = t.Callable):
        layers = []
        if layers_num == 1:
            return nn.Linear(input_size, output_size)
        for i in range(layers_num):
            if i == 0:
                layers.append(nn.Linear(input_size, hidden_size))
                layers.append(nn.LeakyReLU(negative_slope=0.01))
            elif i == layers_num - 1:
                layers.append(nn.BatchNorm1d(hidden_size))
                layers.append(nn.Linear(hidden_size, output_size))
                layers.append(final_activation)
            else:
                layers.append(nn.BatchNorm1d(hidden_size))
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(nn.LeakyReLU(negative_slope=0.01))
        return nn.Sequential(*layers)
    
    def get_noise(self, batch_size: int, device: torch.device, noise_type: str = 'gaussian', **kwargs: t.Dict[str, t.Any]):
        if noise_type == 'gaussian':
            return torch.randn((batch_size, self.nn_in_features), device=device)
        elif noise_type == 'uniform':
            return torch.rand((batch_size, self.nn_in_features), device=device)
        elif noise_type == 'hybrid':
            return torch.cat([torch.randn((batch_size, self.nn_in_features // 2), device=device), torch.rand((batch_size, self.nn_in_features // 2), device=device)], dim=-1)
        elif noise_type == 'custom':
            mean = kwargs['mean'].unsqueeze(0).expand(batch_size, -1)
            std = kwargs['std'].unsqueeze(0).expand(batch_size, -1)
            return torch.normal(mean, std).to(device)
        else:
            raise ValueError('Unknown noise type')


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
        self.block_dec_hidden_size = kwargs.get('block_enc_hidden_size', 128)

        self.seq_dec_depth = kwargs.get('seq_dec_depth', 4)
        self.seq_dec_hidden_size = kwargs.get('seq_dec_hidden_size', 128)
        
        self.activation = kwargs.get('activation', 'relu')

        self.blocks, self.block_pairs = self._initialize_block_pairs()
        self.block_pair_idx_map = self._initialize_block_pair_idx_map()

        self.seq_num = 2*len(self.block_pairs) + self.channel_num - 2

        self.freq_decoder = self._get_mlp(self.freq_dec_depth, self.freq_dim, self.freq_dec_hidden_size, self.freq_dec_hidden_size)
        self.freq_seq_constructor = nn.ModuleList([
            self._get_mlp(self.freq_dec_depth, self.freq_dec_hidden_size, self.freq_dec_hidden_size, self.N, final_activation='sigmoid')
            for _ in range(self.seq_num)
        ])

        self.naive_block_decoder = self._get_mlp(self.block_dec_depth, self.block_dim, self.block_dec_hidden_size, self.seq_num)

        self.naive_seq_decoder = self._get_mlp(self.seq_dec_depth, self.freq_dim+self.block_dim, self.seq_dec_hidden_size, self.N)

        self.tf_seq_decoder_layer = nn.TransformerEncoderLayer(d_model=self.seq_num, nhead=2, dim_feedforward=128, batch_first=True)
        self.tf_seq_decoder = nn.TransformerEncoder(self.tf_seq_decoder_layer, num_layers=1)

        self.lin_mixer = nn.Conv2d(3*self.seq_num, self.seq_num, kernel_size=1, stride=1)


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
        block_expand = block.expand(self.strip_len, -1, -1).permute((1, 2, 0)).unsqueeze(2)

        freq = self.freq_decoder(x[:, :self.freq_dim])
        freq_seq = torch.stack([self._periodic_func(self.freq_seq_constructor[i](freq), i) for i in range(self.kernel_num)], dim=-1)

        tf_input = freq_seq * block_expand.squeeze(2).transpose(1, 2)
        seq = self.tf_seq_decoder(tf_input)
        seq = seq.transpose(1, 2).unsqueeze(2)

        naive_seq = self.naive_seq_decoder(x).unsqueeze(1).unsqueeze(1)

        block_seq = torch.cat([block_expand, seq, naive_seq], dim=1)
        block_seq = self.lin_mixer(block_seq).squeeze(2)

        interaction_block_seq = block_seq[:, :self.channel_num - 2]
        on_site_block_seq = block_seq[:, self.channel_num - 2:]

        H_interaction = torch.stack([self._interaction_block_generator(interaction_block_seq[:, i]) for i in range(self.channel_num - 2)], dim=1)
        H_on_site = torch.stack([self._on_site_block_generator(on_site_block_seq[:, :16]), self._on_site_block_generator(on_site_block_seq[:, 16:])], dim=1)
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
        assert param_vec.shape[1] == 16
        all_blocks = torch.stack([self._block_generator(param_vec[:, i, :], self.blocks[pair[0]], self.blocks[pair[1]]) for i, pair in enumerate(self.block_pairs)], dim=1)
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