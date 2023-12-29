import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


class MultiClassifier(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 1, layers: int = 2, hidden_size: int = 128, classifier_output_idx: int = 0):
        super(MultiClassifier, self).__init__()
        assert input_dim % 2 == 0, f'Input dimension must be even, got {input_dim}'
        self.mlps = nn.ModuleList([self._get_mlp(layers, input_dim, hidden_size, 1) for _ in range(output_dim)])
        self.sigmoid = nn.Sigmoid()
        self.classifier_output_idx = classifier_output_idx
        self.input_dim = input_dim
        self.output_dim = output_dim

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
        x_freq = x[:, :x.shape[1]//2][:, :(self.input_dim * self.output_dim) // 2]
        x_block = x[:, x.shape[1]//2:][:, :(self.input_dim * self.output_dim) // 2]
        # split x into separate parts for each output
        xs_freq = torch.split(x_freq, self.input_dim // 2, dim=1)
        xs_block = torch.split(x_block, self.input_dim // 2, dim=1)
        xs = [torch.cat((x1, x2), dim=1) for x1, x2 in zip(xs_freq, xs_block)]
        output = torch.cat([mlp(x) for mlp, x in zip(self.mlps, xs)], dim=1)
        output[:, self.classifier_output_idx] = self.sigmoid(output[:, self.classifier_output_idx])
        return output


class Classifier(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 1, layers: int = 2, hidden_size: int = 128, classifier_output_idx: int = 0):
        super(Classifier, self).__init__()
        self.mlp = self._get_mlp(layers, input_dim, hidden_size, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.classifier_output_idx = classifier_output_idx
        self.output_dim = output_dim

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
        output = self.mlp(x)
        output[:, self.classifier_output_idx] = self.sigmoid(output[:, self.classifier_output_idx])
        return output


def test_encoder_with_classifier(
    encoder_model: nn.Module,
    classifier_model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
):
    class_criterion = nn.BCELoss()
    reg_criterion = nn.MSELoss(reduction='none')

    encoder_model.to(device)
    classifier_model.to(device)

    encoder_model.eval()
    classifier_model.eval()

    with torch.no_grad():
        total_class_loss = 0
        total_reg_loss = torch.zeros(classifier_model.output_dim - 1).to(device)
        conf_matrix = np.zeros((2, 2))

        for (x, y), _ in tqdm(test_loader, "Testing classifer model"):
            x = x.to(device)
            y = y.to(device)
            z = encoder_model(x)
            output = classifier_model(z)
            all_ids = torch.arange(y.shape[1])
            not_class_ids = all_ids[all_ids != classifier_model.classifier_output_idx]
            prediction = classifier_model(z)
            class_loss = class_criterion(prediction[:, classifier_model.classifier_output_idx], y[:, classifier_model.classifier_output_idx])
            reg_loss = reg_criterion(prediction[:, not_class_ids], y[:, not_class_ids])
            total_class_loss += class_loss.item()
            total_reg_loss += torch.mean(reg_loss, dim=0)

            for i, j in zip(y[:, classifier_model.classifier_output_idx], prediction[:, classifier_model.classifier_output_idx]):
                conf_matrix[int(i.item()), round(j.item())] += 1

        total_class_loss /= len(test_loader)
        total_reg_loss /= len(test_loader)
        sensitivity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
        specifity = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])
        bal_acc = 100.* (sensitivity + specifity) / 2

    print(f'Loss: {total_class_loss}, balanced accuracy: {bal_acc}')

    return total_class_loss, bal_acc, conf_matrix, total_reg_loss.cpu().tolist()


def test_generator_with_classifier(
    generator_model: nn.Module,
    # encoder_model: nn.Module,
    classifier_model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
):
    criterion = nn.BCELoss()

    classifier_model.to(device)
    generator_model.to(device)
    # encoder_model.to(device)

    classifier_model.eval()
    generator_model.eval()
    # encoder_model.eval()

    total_loss = 0
    conf_matrix = np.zeros((2, 2))

    for (x, y), _ in tqdm(test_loader, 'Testing generator for classifier'):
        x = x.to(device)

        z = generator_model.get_noise(x.shape[0], device, noise_type='hybrid')
        x_hat = generator_model.noise_converter(z)
        # latent_prime = encoder_model(x_hat)
        y_hat = classifier_model(x_hat)
        desired_y = torch.ones_like(y_hat)

        loss = criterion(y_hat, desired_y)
        total_loss += loss.item()

        prediction = torch.round(y_hat)

        for i, j in zip(desired_y, prediction):
            conf_matrix[int(i), int(j)] += 1

    total_loss /= len(test_loader)
    specifity = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])
    bal_acc = 100.* specifity

    print(f'Total loss: {total_loss}, balanced accuracy: {bal_acc}')

    return total_loss, bal_acc, conf_matrix


def train_encoder_with_classifier(
    encoder_model: nn.Module,
    decoder_model: nn.Module,
    classifier_model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    epoch: int,
    device: torch.device,
    encoder_optimizer: torch.optim.Optimizer,
    decoder_optimizer: torch.optim.Optimizer,
    classifier_optimizer: torch.optim.Optimizer,
    gt_eigvals: bool = False,
    class_loss_weight: float = 0.01
):

    class_criterion = nn.BCELoss()
    ae_criterion = nn.MSELoss()

    encoder_model.to(device)
    classifier_model.to(device)
    decoder_model.to(device)

    encoder_model.eval()
    classifier_model.train()

    total_loss_class = 0
    total_loss_ae = 0

    print(f'Epoch: {epoch}')
    for (x, y), eig_dec in tqdm(train_loader, 'Training classifier model'):
        x = x.to(device)
        y = y.to(device)
        classifier_optimizer.zero_grad()
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        z = encoder_model(x)

        # reduce the examples so that the number of examples in each class is the same
        # y_sq = y.squeeze()
        # z_reduced = torch.cat((z[y_sq == 0][:len(z[y_sq == 1])], z[y_sq == 1]))
        # y_reduced = torch.cat((y[y_sq == 0][:len(z[y_sq == 1])], y[y_sq == 1]))
        all_ids = torch.arange(y.shape[1])
        not_class_ids = all_ids[all_ids != classifier_model.classifier_output_idx]
        prediction = classifier_model(z)
        loss_class = class_criterion(prediction[:, classifier_model.classifier_output_idx], y[:, classifier_model.classifier_output_idx]) +\
                    ae_criterion(prediction[:, not_class_ids], y[:, not_class_ids])
        total_loss_class += loss_class.item()
        # if len(z_reduced) > 0:
        #     total_loss_class += loss_class.item()

        if gt_eigvals:
            z = (z, eig_dec[0].to(device))
        x_hat = decoder_model(z)
        loss_ae = ae_criterion(x_hat, x)

        total_loss_ae += loss_ae.item()

        loss =class_loss_weight*loss_class + loss_ae
        loss.backward()
        classifier_optimizer.step()
        encoder_optimizer.step()
        decoder_optimizer.step()

    total_loss_ae /= len(train_loader)
    total_loss_class /= len(train_loader)

    print(f'Classification loss: {total_loss_class}\n')
    print(f'Autoencoder loss: {total_loss_ae}\n')

    return total_loss_class, total_loss_ae