import typing as t
import json
import os

import torch
import torch.nn as nn

from models import Classifier, Decoder, DecoderEnsemble, Encoder, EncoderEnsemble, PositionalDecoder, PositionalEncoder, VariationalPositionalEncoder, Generator, Discriminator, EigvalsPositionalDecoder, EigvalsPositionalEncoder

GENERAL_PARAMS_NAME = 'general_params.json'
CLASSIFIER_PARAMS_NAME = 'classifier_params.json'
ENCODER_PARAMS_NAME = 'encoder_params.json'
DECODER_PARAMS_NAME = 'decoder_params.json'
GENERATOR_PARAMS_NAME = 'generator_params.json'
DISCRIMINATOR_PARAMS_NAME = 'discriminator_params.json'
CLASSIFIER_DIR = 'classifier'
ENCODER_DIR = 'encoder'
DECODER_DIR = 'decoder'
GENERATOR_DIR = 'generator'
DISCRIMINATOR_DIR = 'discriminator'
CLASSIFIER_NAME = 'classifier{}.pt'
ENCODER_NAME = 'encoder{}.pt'
DECODER_NAME = 'decoder{}.pt'
GENERATOR_NAME = 'generator{}.pt'
DISCRIMINATOR_NAME = 'discriminator{}.pt'
LATENT_DISTRIBUTION_NAME = 'latent_distribution.txt'
COVARIANCE_MATRIX_NAME = 'covariance_matrix.txt'
DELIMITER = '  '


MODEL_TO_NAMES = {
    Classifier: (CLASSIFIER_PARAMS_NAME, CLASSIFIER_NAME, CLASSIFIER_DIR),
    Decoder: (DECODER_PARAMS_NAME, DECODER_NAME, DECODER_DIR),
    DecoderEnsemble: (DECODER_PARAMS_NAME, DECODER_NAME, DECODER_DIR),
    Encoder: (ENCODER_PARAMS_NAME, ENCODER_NAME, ENCODER_DIR),
    EncoderEnsemble: (ENCODER_PARAMS_NAME, ENCODER_NAME, ENCODER_DIR),
    PositionalDecoder: (DECODER_PARAMS_NAME, DECODER_NAME, DECODER_DIR),
    PositionalEncoder: (ENCODER_PARAMS_NAME, ENCODER_NAME, ENCODER_DIR),
    Generator: (GENERATOR_PARAMS_NAME, GENERATOR_NAME, GENERATOR_DIR),
    Discriminator: (DISCRIMINATOR_PARAMS_NAME, DISCRIMINATOR_NAME, DISCRIMINATOR_DIR),
    VariationalPositionalEncoder: (ENCODER_PARAMS_NAME, ENCODER_NAME, ENCODER_DIR),
    EigvalsPositionalEncoder: (ENCODER_PARAMS_NAME, ENCODER_NAME, ENCODER_DIR),
    EigvalsPositionalDecoder: (DECODER_PARAMS_NAME, DECODER_NAME, DECODER_DIR),
}

def load_autoencoder(root_dir: str, epoch: int) -> t.Tuple[Encoder, Decoder]:
    return load_ae_model(root_dir, epoch, Encoder, Decoder)


def load_autoencoder_ensemble(root_dir: str, epoch: int) -> t.Tuple[EncoderEnsemble, DecoderEnsemble]:
    return load_ae_model(root_dir, epoch, EncoderEnsemble, DecoderEnsemble)


def load_positional_autoencoder(root_dir: str, epoch: int) -> t.Tuple[PositionalEncoder, PositionalDecoder]:
    return load_ae_model(root_dir, epoch, PositionalEncoder, PositionalDecoder)


def load_variational_positional_autoencoder(root_dir: str, epoch: int) -> t.Tuple[VariationalPositionalEncoder, PositionalDecoder]:
    return load_ae_model(root_dir, epoch, VariationalPositionalEncoder, PositionalDecoder)


def load_eigvals_positional_autoencoder(root_dir: str, epoch: int) -> t.Tuple[EigvalsPositionalEncoder, EigvalsPositionalDecoder]:
    return load_ae_model(root_dir, epoch, EigvalsPositionalEncoder, EigvalsPositionalDecoder)


def load_ae_model(root_dir: str, epoch: int, encoder_class: t.Type[nn.Module], decoder_class: t.Type[nn.Module]):
    params, encoder_params, decoder_params = load_autoencoder_params(root_dir, encoder_class, decoder_class)
    encoder_params = get_full_model_config(params, encoder_params)
    decoder_params = get_full_model_config(params, decoder_params)

    encoder = load_model(encoder_class, encoder_params, root_dir, epoch)
    decoder = load_model(decoder_class, decoder_params, root_dir, epoch)

    return encoder, decoder


def load_gan_from_positional_autoencoder(root_dir: str, epoch: int) -> t.Tuple[Generator, Discriminator]:
    return load_gan_model(root_dir, epoch, PositionalDecoder, PositionalEncoder)


def load_gan_model(root_dir: str, epoch: int, sub_generator_class: t.Type[nn.Module], sub_discriminator_class: t.Type[nn.Module]):
    params, generator_config, discriminator_config = load_gan_params(root_dir)
    generator_config = get_full_model_config(params, generator_config)
    discriminator_config = get_full_model_config(params, discriminator_config)
    generator_params = {
        'model_class': sub_generator_class,
        'model_config': generator_config,
    }
    discriminator_params = {
        'model_class': sub_discriminator_class,
        'model_config': discriminator_config,
    }

    generator = load_model(Generator, generator_params, root_dir, epoch)
    discriminator = load_model(Discriminator, discriminator_params, root_dir, epoch)

    return generator, discriminator


def load_generator(root_dir: str, epoch: int, sub_generator_class: t.Type[nn.Module]):
    params, generator_config, _ = load_gan_params(root_dir)
    generator_config = get_full_model_config(params, generator_config)
    generator_params = {
        'model_class': sub_generator_class,
        'model_config': generator_config,
    }

    generator = load_model(Generator, generator_params, root_dir, epoch)
    return generator


def load_gan_submodel_state_dict(root_dir: str, epoch: int, model: t.Union[Generator, Discriminator]):
    model_path = os.path.join(root_dir, MODEL_TO_NAMES[type(model.nn)][2], MODEL_TO_NAMES[type(model.nn)][1].format(f'_ep{epoch}'))
    model.nn.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))


def load_autoencoder_params(root_dir: str, encoder_class: t.Type[nn.Module], decoder_class: t.Type[nn.Module]):
    general_params = load_general_params(root_dir)
    encoder_params = load_model_params(root_dir, encoder_class)    
    decoder_params = load_model_params(root_dir, decoder_class)

    return general_params, encoder_params, decoder_params


def load_gan_params(root_dir: str):
    general_params = load_general_params(root_dir)
    generator_params = load_model_params(root_dir, Generator)
    discriminator_params = load_model_params(root_dir, Discriminator)

    return general_params, generator_params, discriminator_params


def load_general_params(root_dir: str):
    params_save_path = os.path.join(root_dir, GENERAL_PARAMS_NAME)
    general_params = load_params(params_save_path)
    return general_params


def load_model_params(root_dir: str, model_type: t.Type[nn.Module]):
    params_dir = os.path.join(root_dir, MODEL_TO_NAMES[model_type][2])
    params_save_path = os.path.join(params_dir, MODEL_TO_NAMES[model_type][0])
    encoder_params = load_params(params_save_path)
    return encoder_params


def load_params(file_path: str):
    with open(file_path) as f:
        data = json.load(f)
    return data


def get_full_model_config(params: t.Dict[str, t.Any], model_params: t.Dict[str, t.Any]):
    return {
        'representation_dim': params['representation_dim'],
        'input_size': (params['in_channels'], params['N'], params['block_size']),
        'output_size': (params['in_channels'], params['N'], params['block_size']),
        **model_params
    }


def load_classifier(root_dir: str, epoch: int, **classifier_config: t.Dict[str, t.Any]) -> Classifier:
    return load_model(Classifier, classifier_config, root_dir, epoch)


def load_model(model_type: t.Type[nn.Module], model_params: t.Dict[str, t.Any], root_dir: str, epoch: int):
    model = model_type(**model_params)
    model_path = os.path.join(root_dir, MODEL_TO_NAMES[model_type][2], MODEL_TO_NAMES[model_type][1].format(f'_ep{epoch}'))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return model


def save_autoencoder(encoder: torch.nn.Module, decoder: torch.nn.Module, root_dir: str, epoch: int):
    save_model(encoder, root_dir, epoch)
    save_model(decoder, root_dir, epoch)


def save_gan(generator: Generator, discriminator: Discriminator, root_dir: str, epoch: int):
    save_model(generator, root_dir, epoch)
    save_model(discriminator, root_dir, epoch)


def save_model(model: nn.Module, root_dir: str, epoch: int):
    model_type = type(model)
    if not os.path.isdir(root_dir):
        os.makedirs(root_dir)

    model_dir = os.path.join(root_dir, MODEL_TO_NAMES[model_type][2])
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    model_path = os.path.join(model_dir, MODEL_TO_NAMES[model_type][1].format(f'_ep{epoch}'))
    torch.save(model.state_dict(), model_path)


def save_autoencoder_params(
    general_params: t.Dict[str, t.Any],
    encoder_params: t.Dict[str, t.Any],
    decoder_params: t.Dict[str, t.Any],
    root_dir: str,
):
    params_list = [general_params, encoder_params, decoder_params]
    params_dirs = ['', ENCODER_DIR, DECODER_DIR]
    params_save_names = [GENERAL_PARAMS_NAME, ENCODER_PARAMS_NAME, DECODER_PARAMS_NAME]
    save_all_params(params_list, params_dirs, params_save_names, root_dir)


def save_gan_params(
    general_params: t.Dict[str, t.Any],
    generator_params: t.Dict[str, t.Any],
    discriminator_params: t.Dict[str, t.Any],
    root_dir: str,
):
    params_list = [general_params, generator_params, discriminator_params]
    params_dirs = ['', GENERATOR_DIR, DISCRIMINATOR_DIR]
    params_save_names = [GENERAL_PARAMS_NAME, GENERATOR_PARAMS_NAME, DISCRIMINATOR_PARAMS_NAME]
    save_all_params(params_list, params_dirs, params_save_names, root_dir)


def save_all_params(
    params_list: t.List[t.Dict[str, t.Any]],
    params_dirs: t.List[str],
    params_save_names: t.List[str],
    root_dir: str,
):
    for params, params_dir, params_name in zip(params_list, params_dirs, params_save_names):
        params_dir_path = os.path.join(root_dir, params_dir)
        if not os.path.isdir(params_dir_path):
            os.makedirs(params_dir_path)
        params_save_path = os.path.join(params_dir_path, params_name)
        save_params(params, params_save_path)
    

def load_data_list(path: str, skip_header: bool = False):
    with open(path) as f:
        data = f.readlines()
    data = [x.strip() for x in data]
    data = [x.split(DELIMITER) for x in data]
    if skip_header:
        data = data[1:]
    data = [[float(y) for y in x] for x in data]
    return data


def save_data_list(data: t.List[t.Any], path: str, mode:str = 'a'):
    data_str = [str(x) for x in data]
    with open(path, mode) as f:
        f.write(f'{DELIMITER.join(data_str)}\n')


def save_params(params: t.Dict[str, t.Any], file_path: str):
    data = json.dumps(params)
    with open(file_path, 'w') as f:
        f.write(data)


def load_latent_distribution(root_dir: str):
    latent_distribution_path = os.path.join(root_dir, LATENT_DISTRIBUTION_NAME)
    latent_distribution = load_data_list(latent_distribution_path, skip_header=True)
    mean, std = zip(*latent_distribution)
    mean = torch.tensor(mean)
    std = torch.tensor(std)
    return mean, std


def save_latent_distribution(
    latent_distribution: t.Tuple[torch.Tensor, torch.Tensor],
    root_dir: str,
):
    if not os.path.isdir(root_dir):
        os.makedirs(root_dir)

    latent_distribution_path = os.path.join(root_dir, LATENT_DISTRIBUTION_NAME)
    mean, std = latent_distribution
    
    mean = mean.detach().cpu().tolist()
    std = std.detach().cpu().tolist()
    save_data_list(['mean', 'std'], latent_distribution_path, mode='w')
    for mean_i, std_i in zip(mean, std):
        save_data_list([mean_i, std_i], latent_distribution_path)


def save_covariance_matrix(
    covariance_matrix: torch.Tensor,
    root_dir: str,
    prefix: str = ''
):
    if not os.path.isdir(root_dir):
        os.makedirs(root_dir)
    covariance_matrix_path = os.path.join(root_dir, f'{prefix}{COVARIANCE_MATRIX_NAME}')
    covariance_matrix = covariance_matrix.detach().cpu().numpy()
    save_data_list([f'covariance matrix ({len(covariance_matrix)} x {len(covariance_matrix[0])})'], covariance_matrix_path, mode='w')
    for row in covariance_matrix:
        save_data_list(row, covariance_matrix_path, mode='a')


def load_covariance_matrix(root_dir: str, prefix: str = ''):
    covariance_matrix_path = os.path.join(root_dir, f'{prefix}{COVARIANCE_MATRIX_NAME}')
    covariance_matrix = load_data_list(covariance_matrix_path, skip_header=True)
    covariance_matrix = torch.tensor(covariance_matrix)
    return covariance_matrix