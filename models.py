import typing as t

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


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
        '''

        super(Decoder, self).__init__()
        self.channel_num = output_size[0]
        self.N = output_size[1]
        self.block_size = output_size[2]
        self.representation_dim = representation_dim

        self.kernel_size = kwargs.get('kernel_size', self.block_size)
        self.stride = kwargs.get('stride', self.block_size)
        self.dilation = kwargs.get('dilation', 1)
        self.fc_num = kwargs.get('fc_num', 2)
        self.upsample_method = kwargs.get('upsample_method', 'transpose')
        self.conv_num = kwargs.get('conv_num', 1)
        self.kernel_num = kwargs.get('kernel_num', 32)
        self.hidden_size = kwargs.get('hidden_size', 128)
        self.scale_factor = kwargs.get('scale_factor', 2*self.stride)
        self.convs_input_size = self._get_convs_input_size()
        self.fcs_output_size = (self._get_convs_input_size() ** 2) * self.kernel_num

        self.fcs = self._get_fcs()
        self.convs = self._get_convs()


    def _get_convs(self):
        convs = []
        if self.upsample_method == 'transpose':
            for _ in range(1, self.conv_num):
                convs.append(nn.ConvTranspose2d(self.kernel_num, self.kernel_num, kernel_size=self.kernel_size, stride=self.stride, dilation=self.dilation))
                convs.append(nn.ReLU())
                convs.append(nn.BatchNorm2d(self.kernel_num))
            convs.append(nn.ConvTranspose2d(self.kernel_num, self.channel_num, kernel_size=self.kernel_size, stride=self.stride, dilation=self.dilation))
        else:
            for _ in range(1, self.conv_num):
                convs.append(nn.Upsample(scale_factor=self.stride, mode=self.upsample_method))

                convs.append(nn.Conv2d(self.kernel_num, self.kernel_num, kernel_size=self.kernel_size, stride=self.stride, dilation=self.dilation))
                convs.append(nn.ReLU())
                convs.append(nn.BatchNorm2d(self.kernel_num))
            convs.append(nn.Conv2d(self.kernel_num, self.channel_num, kernel_size=self.kernel_size, stride=self.stride, dilation=self.dilation))
            convs.append(nn.Upsample(scale_factor=self.stride, mode=self.upsample_method))
        return nn.Sequential(*convs)


    def _get_convs_input_size(self):
        if self.upsample_method == 'transpose':
            input_size = (self.N*self.block_size - self.dilation*(self.kernel_size-1) - 1) // self.stride + 1
            for _ in range(1, self.conv_num):
                input_size = (input_size - self.dilation*(self.kernel_size-1) - 1) // self.stride + 1
        else:
            input_size = ((self.N*self.block_size - 1) * self.stride + self.dilation*(self.kernel_size-1) + 1) // self.scale_factor
            for _ in range(1, self.conv_num):
                input_size = ((input_size - 1) * self.stride + self.dilation*(self.kernel_size-1) + 1) // self.scale_factor
        return input_size

    
    def _get_fcs(self):
        fcs = []
        fcs.append(nn.Linear(self.representation_dim, self.hidden_size))
        fcs.append(nn.ReLU())
        fcs.append(nn.BatchNorm1d(self.hidden_size))
        for _ in range(1, self.fc_num - 1):
            fcs.append(nn.Linear(self.hidden_size, self.hidden_size))
            fcs.append(nn.ReLU())
            fcs.append(nn.BatchNorm1d(self.hidden_size))
        fcs.append(nn.Linear(self.hidden_size, self.fcs_output_size))
        return nn.Sequential(*fcs)

    
    def forward(self, x: torch.Tensor):
        x = self.fcs(x)
        x = x.view(-1, self.kernel_num, self.convs_input_size, self.convs_input_size)
        x = self.convs(x)
        return x


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
        '''
        super(Encoder, self).__init__()
        self.channel_num = input_size[0]
        self.N = input_size[1]
        self.block_size = input_size[2]
        self.representation_dim = representation_dim

        self.kernel_size = kwargs.get('kernel_size', self.block_size)
        self.stride = kwargs.get('stride', self.block_size)
        self.dilation = kwargs.get('dilation', 1)
        self.fc_num = kwargs.get('fc_num', 2)
        self.conv_num = kwargs.get('conv_num', 1)
        self.kernel_num = kwargs.get('kernel_num', 32)
        self.hidden_size = kwargs.get('hidden_size', 128)
        self.convs_output_size = (self._get_convs_output_size() ** 2) * self.kernel_num
        
        self.convs = self._get_convs()
        self.fcs = self._get_fcs()


    def _get_convs(self):
        convs = []
        convs.append(nn.Conv2d(self.channel_num, self.kernel_num, kernel_size= self.kernel_size, stride=self.stride, dilation=self.dilation))
        convs.append(nn.ReLU())
        convs.append(nn.BatchNorm2d(self.kernel_num))
        for _ in range(1, self.conv_num):
            convs.append(nn.Conv2d(self.kernel_num, self.kernel_num, kernel_size=self.kernel_size, stride=self.stride, dilation=self.dilation))
            convs.append(nn.ReLU())
            convs.append(nn.BatchNorm2d(self.kernel_num))
        return nn.Sequential(*convs)


    def _get_convs_output_size(self):
        output_size = (self.N*self.block_size - self.dilation*(self.kernel_size-1) - 1) // self.stride + 1
        for _ in range(1, self.conv_num):
            output_size = (output_size - self.dilation*(self.kernel_size-1) - 1) // self.stride + 1
        return output_size

    
    def _get_fcs(self):
        fcs = []
        fcs.append(nn.Linear(self.convs_output_size, self.hidden_size))
        fcs.append(nn.ReLU())
        fcs.append(nn.BatchNorm1d(self.hidden_size))
        for _ in range(1, self.fc_num - 1):
            fcs.append(nn.Linear(self.hidden_size, self.hidden_size))
            fcs.append(nn.ReLU())
            fcs.append(nn.BatchNorm1d(self.hidden_size))
        fcs.append(nn.Linear(self.hidden_size, self.representation_dim))
        return nn.Sequential(*fcs)

    
    def forward(self, x: torch.Tensor):
        x = self.convs(x)
        x = x.view(-1, self.convs_output_size)
        x = self.fcs(x)
        return x


def reconstruct_hamiltonian(H: np.ndarray, encoder: Encoder, decoder: Decoder):
    H_torch = torch.from_numpy(H)
    H_torch = torch.stack((H_torch.real, H_torch.imag), dim= 0)
    H_torch = H_torch.unsqueeze(0)

    latent_vec = encoder(H_torch)
    H_torch_rec = decoder(latent_vec)
    H_rec = torch.complex(H_torch_rec[:, 0, :, :], H_torch_rec[:, 1, :, :]).squeeze().numpy()
    return H_rec


def site_perm(x: torch.Tensor, N: int, block_size: int):
    permutation = torch.randperm(N)
    permuted_indices = torch.cat([torch.arange(i*block_size, (i+1)*block_size) for i in permutation], dim=0)
    x_permuted = x[:, :, permuted_indices, :][:, :, :, permuted_indices]
    return x_permuted


def test_autoencoder(
    encoder_model: nn.Module,
    decoder_model: nn.Module,
    test_loader: torch.utils.data.DataLoader, 
    device: torch.device, 
    site_permutation: bool = False,
):
    criterion = nn.MSELoss()

    encoder_model.to(device)
    decoder_model.to(device)

    encoder_model.eval()
    decoder_model.eval()

    total_loss = 0

    for x, _ in tqdm(test_loader, "Testing model"):
        x = x.to(device)
        if site_permutation:
            x = site_perm(x, encoder_model.N, encoder_model.block_size)
        z = encoder_model(x)
        x_hat = decoder_model(z)
        loss = criterion(x_hat, x)

        total_loss += loss.item()
    
    total_loss /= len(test_loader)
    print(f'Loss: {total_loss}\n')

    return total_loss


def train_autoencoder(
    encoder_model: nn.Module,
    decoder_model: nn.Module,
    train_loader: torch.utils.data.DataLoader, 
    epoch: int,
    device: torch.device, 
    encoder_optimizer: torch.optim.Optimizer,
    decoder_optimizer: torch.optim.Optimizer,
    site_permutation: bool = False,
):
    criterion = nn.MSELoss()

    encoder_model.to(device)
    decoder_model.to(device)

    encoder_model.train()
    decoder_model.train()

    total_loss = 0

    print(f'Epoch: {epoch}')
    for x, _ in tqdm(train_loader, 'Training model'):
        x = x.to(device)
        if site_permutation:
            x = site_perm(x, encoder_model.N, encoder_model.block_size)
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        z = encoder_model(x)
        x_hat = decoder_model(z)
        loss = criterion(x_hat, x)
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()
    
    total_loss /= len(train_loader)
    print(f'Loss: {total_loss}\n')

    return total_loss
