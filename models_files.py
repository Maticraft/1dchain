import typing as t
import json
import os

import torch
import torch.nn as nn

from models import Decoder, DecoderEnsemble, Encoder, EncoderEnsemble, PositionalDecoder, PositionalEncoder

GENERAL_PARAMS_NAME = 'general_params.json'
ENCODER_PARAMS_NAME = 'encoder_params.json'
DECODER_PARAMS_NAME = 'decoder_params.json'
ENCODER_DIR = 'encoder'
DECODER_DIR = 'decoder'
ENCODER_NAME = 'encoder{}.pt'
DECODER_NAME = 'decoder{}.pt'
DELIMITER = '  '


def load_ae_model(root_dir: str, epoch: int, encoder_class: nn.Module, decoder_class: nn.Module):
    params, encoder_params, decoder_params = load_autoencoder_params(root_dir)

    encoder = encoder_class((params['in_channels'], params['N'], params['block_size']), params['representation_dim'], **encoder_params)
    encoder_path = os.path.join(root_dir, ENCODER_DIR, ENCODER_NAME.format(f'_ep{epoch}'))
    encoder.load_state_dict(torch.load(encoder_path))

    decoder = decoder_class(params['representation_dim'], (params['in_channels'], params['N'], params['block_size']), **decoder_params)
    decoder_path = os.path.join(root_dir, DECODER_DIR, DECODER_NAME.format(f'_ep{epoch}'))
    decoder.load_state_dict(torch.load(decoder_path))

    return encoder, decoder


def load_autoencoder(root_dir: str, epoch: int):
    return load_ae_model(root_dir, epoch, Encoder, Decoder)


def load_autoencoder_ensemble(root_dir: str, epoch: int):
    return load_ae_model(root_dir, epoch, EncoderEnsemble, DecoderEnsemble)


def load_autoencoder_params(root_dir: str):
    # Load general params
    params_save_path = os.path.join(root_dir, GENERAL_PARAMS_NAME)
    general_params = load_params(params_save_path)

    # Load encoder params
    params_dir = os.path.join(root_dir, ENCODER_DIR)
    params_save_path = os.path.join(params_dir, ENCODER_PARAMS_NAME)
    encoder_params = load_params(params_save_path)
    
    # Save decoder params
    params_dir = os.path.join(root_dir, DECODER_DIR)
    params_save_path = os.path.join(params_dir, DECODER_PARAMS_NAME)
    decoder_params = load_params(params_save_path)

    return general_params, encoder_params, decoder_params


def load_params(file_path: str):
    with open(file_path) as f:
        data = json.load(f)
    return data


def load_positional_autoencoder(root_dir: str, epoch: int):
    return load_ae_model(root_dir, epoch, PositionalEncoder, PositionalDecoder)


def save_autoencoder(encoder: torch.nn.Module, decoder: torch.nn.Module, root_dir: str, epoch: int):
    if not os.path.isdir(root_dir):
        os.makedirs(root_dir)

    encoder_path = os.path.join(root_dir, ENCODER_DIR, ENCODER_NAME.format(f'_ep{epoch}'))
    decoder_path = os.path.join(root_dir, DECODER_DIR, DECODER_NAME.format(f'_ep{epoch}'))

    torch.save(encoder.state_dict(), encoder_path)
    torch.save(decoder.state_dict(), decoder_path)


def save_autoencoder_params(
    general_params: t.Dict[str, t.Any],
    encoder_params: t.Dict[str, t.Any],
    decoder_params: t.Dict[str, t.Any],
    root_dir: str,
):
    if not os.path.isdir(root_dir):
        os.makedirs(root_dir)
    
    # Save general params
    params_save_path = os.path.join(root_dir, GENERAL_PARAMS_NAME)
    save_params(general_params, params_save_path)

    # Save encoder params
    params_dir = os.path.join(root_dir, ENCODER_DIR)
    if not os.path.isdir(params_dir):
        os.makedirs(params_dir)
    params_save_path = os.path.join(params_dir, ENCODER_PARAMS_NAME)
    save_params(encoder_params, params_save_path)
    
    # Save decoder params
    params_dir = os.path.join(root_dir, DECODER_DIR)
    if not os.path.isdir(params_dir):
        os.makedirs(params_dir)
    params_save_path = os.path.join(params_dir, DECODER_PARAMS_NAME)
    save_params(decoder_params, params_save_path)


def save_data_list(data: t.List[t.Any], path: str, mode:str = 'a'):
    data_str = [str(x) for x in data]
    with open(path, mode) as f:
        f.write(f'{DELIMITER.join(data_str)}\n')


def save_params(params: t.Dict[str, t.Any], file_path: str):
    data = json.dumps(params)
    with open(file_path, 'w') as f:
        f.write(data)