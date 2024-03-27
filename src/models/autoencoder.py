import typing as t
from tqdm import tqdm

import torch
import torch.nn as nn

from src.models.utils import determinant_loss, diagonal_loss, edge_diff, eigenvectors_loss, log_scale_loss, site_perm



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
                convs.append(nn.BatchNorm2d(self.kernel_num))
                convs.append(self._get_activation())
            if self.conv_num > 1:
                convs.append(nn.ConvTranspose2d(self.kernel_num, self.kernel_num1, kernel_size=self.kernel_size, stride=self.stride, dilation=self.dilation))
                convs.append(nn.BatchNorm2d(self.kernel_num1))
                convs.append(self._get_activation())

            convs.append(nn.ConvTranspose2d(self.kernel_num1, self.channel_num, kernel_size=self.kernel_size1, stride=self.stride1, dilation=self.dilation1))
        else:
            for i in range(1, self.conv_num - 1):
                if self.stride > 1:
                    raise ValueError("Upsample not implemented for stride larger than 1")
                convs.append(nn.Upsample(scale_factor=self.scale_factor[-i], mode=self.upsample_method))
                convs.append(nn.Conv2d(self.kernel_num, self.kernel_num, kernel_size=self.kernel_size, stride=self.stride, dilation=self.dilation, padding='same'))
                convs.append(nn.BatchNorm2d(self.kernel_num))
                convs.append(self._get_activation())
            if self.conv_num > 1:
                convs.append(nn.Upsample(scale_factor=self.scale_factor[1], mode=self.upsample_method))
                convs.append(nn.Conv2d(self.kernel_num, self.kernel_num1, kernel_size=self.kernel_size, stride=self.stride, dilation=self.dilation, padding='same'))
                convs.append(nn.BatchNorm2d(self.kernel_num1))
                convs.append(self._get_activation())
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
        fcs.append(nn.BatchNorm1d(self.hidden_size))
        fcs.append(self._get_activation())
        for _ in range(1, self.fc_num - 1):
            fcs.append(nn.Linear(self.hidden_size, self.hidden_size))
            fcs.append(nn.BatchNorm1d(self.hidden_size))
            fcs.append(self._get_activation())
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

    def forward(self, x: torch.Tensor):
        x = self.fcs(x)
        x = x.view(-1, self.kernel_num, self.convs_input_size[0], self.convs_input_size[1])
        x = self.convs(x)
        if self.use_strips:
            x = self._get_matrix_from_strips(x)
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
        convs.append(nn.BatchNorm2d(self.kernel_num1))
        convs.append(self._get_activation())

        if self.conv_num > 1:
            convs.append(nn.Conv2d(self.kernel_num1, self.kernel_num, kernel_size= self.kernel_size, stride=self.stride, dilation=self.dilation))
            convs.append(nn.BatchNorm2d(self.kernel_num))
            convs.append(self._get_activation())

        for _ in range(2, self.conv_num):
            convs.append(nn.Conv2d(self.kernel_num, self.kernel_num, kernel_size=self.kernel_size, stride=self.stride, dilation=self.dilation))
            convs.append(nn.BatchNorm2d(self.kernel_num))
            convs.append(self._get_activation())
        return nn.Sequential(*convs)

    def _get_convs_output_size(self, dim):
        output_size = (self.site_size[dim]*self.block_size - self.dilation1[dim]*(self.kernel_size1[dim]-1) - 1) // self.stride1[dim] + 1
        for _ in range(1, self.conv_num):
            output_size = (output_size - self.dilation[dim]*(self.kernel_size[dim]-1) - 1) // self.stride[dim] + 1
        return output_size

    def _get_fcs(self):
        fcs = []
        fcs.append(nn.Linear(self.convs_output_size, self.hidden_size))
        fcs.append(nn.BatchNorm1d(self.hidden_size))
        fcs.append(self._get_activation())
        for _ in range(1, self.fc_num - 1):
            fcs.append(nn.Linear(self.hidden_size, self.hidden_size))
            fcs.append(nn.BatchNorm1d(self.hidden_size))
            fcs.append(self._get_activation())
        fcs.append(nn.Linear(self.hidden_size, self.representation_dim))
        return nn.Sequential(*fcs)

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
        if self.use_strips:
            strip_bound = ((self.channel_num // 2) - 1) // 2
            x = torch.cat([self._get_strip(x, i, fill_mode='hamiltonian') for i in range(-strip_bound, strip_bound + 1)], dim=1)
        x = self.convs(x)
        x = x.view(-1, self.convs_output_size)
        x = self.fcs(x)
        return x


def test_autoencoder(
    encoder_model: nn.Module,
    decoder_model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    site_permutation: bool = False,
    edge_loss: bool = False,
    eigenvalues_loss: bool = False,
    eigenstates_loss: bool = False,
    diag_loss: bool = False,
    det_loss: bool = False,
    gt_eigvals = False,
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
    total_eigenvalues_loss = 0
    total_eigenstates_loss = 0
    total_diag_loss = 0
    total_det_loss = 0

    with torch.no_grad():
        for (x, _), eig_dec in tqdm(test_loader, "Testing autoencoder model"):
            x = x.to(device)
            if site_permutation:
                x = site_perm(x, encoder_model.N, encoder_model.block_size)
            z = encoder_model(x)
            if gt_eigvals:
                z = (z, eig_dec[0].to(device))
            x_hat = decoder_model(z)
            loss = criterion(x_hat, x)
            total_loss += loss.item()

            if edge_loss:
                e_loss = edge_diff(x_hat, x, criterion, edge_width=8)
                total_edge_loss += e_loss.item()

            if eigenvalues_loss:
                assert eig_dec is not None, "Incorrect eigen decomposition values"
                target_eigvals = eig_dec[0].to(device)
                if isinstance(z, tuple):
                    encoded_eigvals = z[1]
                else:
                    encoded_eigvals = torch.linalg.eigvals(z).real
                eigvals_loss = criterion(encoded_eigvals, target_eigvals)
                total_eigenvalues_loss += eigvals_loss.item()

            if det_loss:
                assert eig_dec is not None, "Incorrect eigen decomposition values"
                target_eigvals = eig_dec[0].to(device)
                det_loss = determinant_loss(x_hat, target_eigvals, criterion)
                total_det_loss += det_loss.item()

            if eigenstates_loss:
                assert eig_dec is not None, "Incorrect eigen decomposition values"
                eig_dec = eig_dec[0].to(device), eig_dec[1].to(device)
                eig_loss = eigenvectors_loss(x_hat, eig_dec, criterion)
                total_eigenstates_loss += eig_loss.item()

            if diag_loss:
                diag_loss = diagonal_loss(x_hat, x, criterion, block_size=4)
                total_diag_loss += diag_loss.item()

    total_loss /= len(test_loader)
    total_edge_loss /= len(test_loader)
    total_eigenvalues_loss /= len(test_loader)
    total_eigenstates_loss /= len(test_loader)
    total_diag_loss /= len(test_loader)
    total_det_loss /= len(test_loader)

    print(f'Loss: {total_loss}')
    if edge_loss:
        print(f'Edge Loss: {total_edge_loss}')
    if eigenstates_loss:
        print(f'Eigenstates Loss: {total_eigenstates_loss}')
    print()

    return total_loss, total_edge_loss, total_eigenvalues_loss, total_eigenstates_loss, total_diag_loss, total_det_loss


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
    eigenvalues_loss: bool = False,
    eigenvalues_loss_weight: float = .5,
    eigenstates_loss: bool = False,
    eigenstates_loss_weight: float = .5,
    diag_loss: bool = False,
    diag_loss_weight: float = .01,
    det_loss: bool = False,
    det_loss_weight: float = .01,
    log_scaled_loss: bool = False,
    gt_eigvals = False,
):
    if site_permutation and edge_loss:
        raise NotImplementedError("Combining edge loss with site permutation is not implemented")

    if log_scaled_loss:
        assert not diag_loss, "Diagonal loss is not implemented for log scaled loss"
        criterion = nn.MSELoss(reduction='none')
    else:
        criterion = nn.MSELoss()

    encoder_model.to(device)
    decoder_model.to(device)

    encoder_model.train()
    decoder_model.train()

    total_loss = 0
    total_edge_loss = 0
    total_eigenvalues_loss = 0
    total_eigenstates_loss = 0
    total_diag_loss = 0
    total_det_loss = 0

    print(f'Epoch: {epoch}')
    for (x, _), eig_dec in tqdm(train_loader, 'Training autoencoder model'):
        x = x.to(device)
        if site_permutation:
            x = site_perm(x, encoder_model.N, encoder_model.block_size)
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        z = encoder_model(x)
        if gt_eigvals:
            z = (z, eig_dec[0].to(device))
        x_hat = decoder_model(z)
        loss = criterion(x_hat, x)
        total_loss += torch.mean(loss).item()

        if edge_loss:
            e_loss = edge_diff(x_hat, x, criterion, edge_width=8)
            loss += edge_loss_weight * e_loss
            total_edge_loss += torch.mean(e_loss).item()

        if eigenvalues_loss:
            assert eig_dec is not None, "Incorrect eigen decomposition values"
            if isinstance(z, tuple):
                encoded_eigvals = z[1]
            else:
                encoded_eigvals = torch.linalg.eigvals(x_hat)
            target_eigvals = eig_dec[0].to(device)
            eig_loss = criterion(encoded_eigvals, target_eigvals)
            loss += eigenvalues_loss_weight * eig_loss
            total_eigenvalues_loss += torch.mean(eig_loss).item()

        if det_loss:
            assert eig_dec is not None, "Incorrect eigen decomposition values"
            target_eigvals = eig_dec[0].to(device)
            det_loss = determinant_loss(x_hat, target_eigvals, criterion)
            loss += det_loss_weight * det_loss
            total_det_loss += torch.mean(det_loss).item()

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

        if log_scaled_loss:
            loss = log_scale_loss(x, loss)
            loss = torch.mean(loss)
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()


    total_loss /= len(train_loader)
    total_edge_loss /= len(train_loader)
    total_eigenvalues_loss /= len(train_loader)
    total_eigenstates_loss /= len(train_loader)
    total_diag_loss /= len(train_loader)
    total_det_loss /= len(train_loader)

    print(f'Loss: {total_loss}')
    if edge_loss:
        print(f'Edge Loss: {total_edge_loss}')
    if eigenstates_loss:
        print(f'Eigenstates Loss: {total_eigenstates_loss}')
    print()

    return total_loss, total_edge_loss, total_eigenvalues_loss, total_eigenstates_loss, total_diag_loss, total_det_loss