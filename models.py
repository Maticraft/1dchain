import typing as t

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


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
        self.kernel_size = kwargs.get('kernel_size', self.block_size)
        self.kernel_size1 = kwargs.get('kernel_size1', self.kernel_size)
        self.stride = kwargs.get('stride', self.block_size)
        self.stride1 = kwargs.get('stride1', self.stride)
        self.dilation = kwargs.get('dilation', 1)
        self.dilation1 = kwargs.get('dilation1', self.dilation)
        self.kernel_num = kwargs.get('kernel_num', 32)
        self.kernel_num1 = kwargs.get('kernel_num1', self.kernel_num)
        self.activation = kwargs.get('activation', 'relu')

        sf = kwargs.get('scale_factor', 2*self.stride)
        self.scale_factor = self._get_scale_factor(sf)

        self.convs_input_size = int(self._get_convs_input_size())

        if self.conv_num > 1:
            self.fcs_output_size = (self.convs_input_size ** 2) * self.kernel_num
        else:
            self.fcs_output_size = (self.convs_input_size ** 2) * self.kernel_num1

        self.fcs = self._get_fcs()
        self.convs = self._get_convs()

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

    def _get_convs_input_size(self):
        if self.upsample_method == 'transpose':
            input_size = (self.N*self.block_size - self.dilation1*(self.kernel_size1-1) - 1) // self.stride1 + 1
            for _ in range(1, self.conv_num):
                input_size = (input_size - self.dilation*(self.kernel_size-1) - 1) // self.stride + 1
        else:
            input_size = round(self.N*self.block_size / self.scale_factor[0])
            for i in range(1, self.conv_num):
                input_size = round(input_size / self.scale_factor[i])
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

    def _get_scale_factor(self, sf: t.Union[float, t.List[float]]):
        if type(sf) == float or type(sf) == int:
            return [sf for _ in range(self.conv_num)]
        elif type(sf) == list:
            if len(sf) != self.conv_num:
                raise ValueError("Wrong number of scale factors")
            return sf
        else:
            raise TypeError("Wrong type of scale factor: must be either int, float or list")

    def forward(self, x: torch.Tensor):
        x = self.fcs(x)
        x = x.view(-1, self.kernel_num, self.convs_input_size, self.convs_input_size)
        x = self.convs(x)
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
        '''
        super(Encoder, self).__init__()
        self.channel_num = input_size[0]
        self.N = input_size[1]
        self.block_size = input_size[2]
        self.representation_dim = representation_dim

        self.kernel_size = kwargs.get('kernel_size', self.block_size)
        self.kernel_size1 = kwargs.get('kernel_size1', self.kernel_size)
        self.stride = kwargs.get('stride', self.block_size)
        self.stride1 = kwargs.get('stride1', self.stride)
        self.dilation = kwargs.get('dilation', 1)
        self.dilation1 = kwargs.get('dilation1', self.dilation)
        self.fc_num = kwargs.get('fc_num', 2)
        self.conv_num = kwargs.get('conv_num', 1)
        self.kernel_num = kwargs.get('kernel_num', 32)
        self.kernel_num1 = kwargs.get('kernel_num1', self.kernel_num)
        self.hidden_size = kwargs.get('hidden_size', 128)
        self.activation = kwargs.get('activation', 'relu')

        if self.conv_num > 1:
            self.convs_output_size = (self._get_convs_output_size() ** 2) * self.kernel_num
        else:
            self.convs_output_size = (self._get_convs_output_size() ** 2) * self.kernel_num1

        self.convs = self._get_convs()
        self.fcs = self._get_fcs()

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

    def _get_convs_output_size(self):
        output_size = (self.N*self.block_size - self.dilation1*(self.kernel_size1-1) - 1) // self.stride1 + 1
        for _ in range(1, self.conv_num):
            output_size = (output_size - self.dilation*(self.kernel_size-1) - 1) // self.stride + 1
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
    
    def forward(self, x: torch.Tensor):
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


def reconstruct_hamiltonian(H: np.ndarray, encoder: nn.Module, decoder: nn.Module, device: torch.device = torch.device('cpu')):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        H_torch = torch.from_numpy(H)
        H_torch = torch.stack((H_torch.real, H_torch.imag), dim= 0)
        H_torch = H_torch.unsqueeze(0).float().to(device)

        latent_vec = encoder(H_torch)
        H_torch_rec = decoder(latent_vec)
        H_rec = torch.complex(H_torch_rec[:, 0, :, :], H_torch_rec[:, 1, :, :]).squeeze().cpu().numpy()
    return H_rec


def edge_diff(x_hat: torch.Tensor, x: torch.Tensor, criterion: t.Callable, edge_width: int = 4):
    x_hat_edges = get_edges(x_hat, edge_width)
    x_edges = get_edges(x, edge_width)
    diff = criterion(x_hat_edges[x_hat_edges != 0.], x_edges[x_hat_edges != 0.])
    return diff


def get_edges(x: torch.Tensor, edge_width: int):
    edges = torch.zeros_like(x)
    edges[:, :, :edge_width, :] = x[:, :, :edge_width, :]
    edges[:, :, -edge_width:, :] = x[:, :, -edge_width:, :]
    edges[:, :, :, :edge_width] = x[:, :, :, :edge_width]
    edges[:, :, :, -edge_width:] = x[:, :, :, -edge_width:]
    return edges


def site_perm(x: torch.Tensor, N: int, block_size: int):
    permutation = torch.randperm(N)
    permuted_indices = torch.cat([torch.arange(i*block_size, (i+1)*block_size) for i in permutation], dim=0)
    x_permuted = x[:, :, permuted_indices, :][:, :, :, permuted_indices]
    return x_permuted


def eigenvectors_loss(x_hat: torch.Tensor, x: torch.Tensor, criterion: t.Callable):
    assert x_hat.shape[1] == 2 and x.shape[1] == 2, 'Wrong dimension of complex tensors'
    x_complex = torch.complex(x[:, 0, :, :], x[:, 1, :, :])
    x_hat_complex = torch.complex(x_hat[:, 0, :, :], x_hat[:, 1, :, :])
    eigvals, eigvec = torch.linalg.eigh(x_complex)
    eigvals_hat, eigvec_hat = torch.linalg.eigh(x_hat_complex)
    eigvec = torch.stack((torch.real(eigvec), torch.imag(eigvec)), dim=1)
    eigvec_hat = torch.stack((torch.real(eigvec_hat), torch.imag(eigvec_hat)), dim=1)
    return criterion(eigvec_hat, eigvec)


def test_autoencoder(
    encoder_model: nn.Module,
    decoder_model: nn.Module,
    test_loader: torch.utils.data.DataLoader, 
    device: torch.device, 
    site_permutation: bool = False,
    edge_loss: bool = False,
    eigenstates_loss: bool = False,
):
    if site_permutation and edge_loss:
        raise NotImplementedError("Combining edge loss with site permutation is not implemented")

    criterion = nn.MSELoss()

    encoder_model.to(device)
    decoder_model.to(device)

    encoder_model.eval()
    decoder_model.eval()

    total_loss = 0
    total_edge_loss = 0
    total_eigenstates_loss = 0

    for x, _ in tqdm(test_loader, "Testing autoencoder model"):
        x = x.to(device)
        if site_permutation:
            x = site_perm(x, encoder_model.N, encoder_model.block_size)
        z = encoder_model(x)
        x_hat = decoder_model(z)
        loss = criterion(x_hat, x)
        total_loss += loss.item()

        if edge_loss:
            e_loss = edge_diff(x_hat, x, criterion, edge_width=8)
            total_edge_loss += e_loss.item()

        if eigenstates_loss:
            eig_loss = eigenvectors_loss(x_hat, x, criterion)
            total_eigenstates_loss += eig_loss.item()

    total_loss /= len(test_loader)
    total_edge_loss /= len(test_loader)
    total_eigenstates_loss /= len(test_loader)

    print(f'Loss: {total_loss}')
    if edge_loss:
        print(f'Edge Loss: {total_edge_loss}')
    if eigenstates_loss:
        print(f'Eigenstates Loss: {total_eigenstates_loss}')
    print()

    return total_loss, total_edge_loss, total_eigenstates_loss


def test_classifier(
    encoder_model: nn.Module,
    classifier_model: nn.Module,
    test_loader: torch.utils.data.DataLoader, 
    device: torch.device, 
):
    criterion = nn.BCELoss()

    encoder_model.to(device)
    classifier_model.to(device)

    encoder_model.eval()
    classifier_model.eval()

    total_loss = 0
    conf_matrix = np.zeros((2, 2))

    for x, y in tqdm(test_loader, "Testing classifer model"):
        x = x.to(device)
        z = encoder_model(x)
        output = classifier_model(z)
        loss = criterion(prediction, y)
        total_loss += loss.item()

        prediction = torch.round(output)              

        for i, j in zip(y, prediction):
            conf_matrix[int(i), int(j)] += 1

    total_loss /= len(test_loader)
    sensitivity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
    specifity = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])
    bal_acc = 100.* (sensitivity + specifity) / 2

    print(f'Loss: {total_loss}, balanced accuracy: {bal_acc}')

    return total_loss, bal_acc, conf_matrix


def train_autoencoder(
    encoder_model: nn.Module,
    decoder_model: nn.Module,
    train_loader: torch.utils.data.DataLoader, 
    epoch: int,
    device: torch.device, 
    encoder_optimizer: torch.optim.Optimizer,
    decoder_optimizer: torch.optim.Optimizer,
    site_permutation: bool = False,
    edge_loss: bool = False,
    edge_loss_weight: float = .5,
    eigenstates_loss: bool = False,
    eigenstates_loss_weight: float = .5
):
    if site_permutation and edge_loss:
        raise NotImplementedError("Combining edge loss with site permutation is not implemented")

    criterion = nn.MSELoss()

    encoder_model.to(device)
    decoder_model.to(device)

    encoder_model.train()
    decoder_model.train()

    total_loss = 0
    total_edge_loss = 0
    total_eigenstates_loss = 0

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
        total_loss += loss.item()

        if edge_loss:
            e_loss = edge_diff(x_hat, x, criterion, edge_width=8)
            loss += edge_loss_weight * e_loss
            total_edge_loss += e_loss.item()

        if eigenstates_loss:
            eig_loss = eigenvectors_loss(x_hat, x, criterion)
            loss += eigenstates_loss_weight * eig_loss
            total_eigenstates_loss += eig_loss.item()

        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

    
    total_loss /= len(train_loader)
    total_edge_loss /= len(train_loader)
    total_eigenstates_loss /= len(train_loader)

    print(f'Loss: {total_loss}')
    if edge_loss:
        print(f'Edge Loss: {total_edge_loss}')
    if eigenstates_loss:
        print(f'Eigenstates Loss: {total_eigenstates_loss}')
    print()

    return total_loss, total_edge_loss, total_eigenstates_loss


def train_classifier(
    encoder_model: nn.Module,
    classifier_model: nn.Module,
    train_loader: torch.utils.data.DataLoader, 
    epoch: int,
    device: torch.device, 
    optimizer: torch.optim.Optimizer,
):

    criterion = nn.BCELoss()

    encoder_model.to(device)
    classifier_model.to(device)

    encoder_model.eval()
    classifier_model.train()

    total_loss = 0

    print(f'Epoch: {epoch}')
    for x, y in tqdm(train_loader, 'Training model'):
        x = x.to(device)
        optimizer.zero_grad()

        z = encoder_model(x).detach()

        prediction = classifier_model(z)
        loss = criterion(prediction, y)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
    
    total_loss /= len(train_loader)

    print(f'Loss: {total_loss}\n')

    return total_loss
