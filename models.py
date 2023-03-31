import typing as t

import torch
import torch.nn as nn

from models_utils import get_edges

class Classifier(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 1, layers: int = 2, hidden_size: int = 128):
        self.mlp = self._get_mlp(layers, input_dim, hidden_size, output_dim)
        self.sigmoid = nn.Sigmoid()

    def _get_mlp(self, layers: int, input_dim: int, hidden_size: int, output_dim: int):
        mlp = []
        mlp.append(nn.Linear(input_dim, hidden_size))
        mlp.append(nn.ReLU)
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
