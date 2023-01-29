import typing as t
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from models import Encoder, Decoder

GENERAL_PARAMS_NAME = 'general_params.json'
ENCODER_PARAMS_NAME = 'encoder_params.json'
DECODER_PARAMS_NAME = 'decoder_params.json'
ENCODER_DIR = 'encoder'
DECODER_DIR = 'decoder'
ENCODER_NAME = 'encoder{}.pt'
DECODER_NAME = 'decoder{}.pt'
DELIMITER = '  '

def save_params(params: t.Dict[str, t.Any], file_path: str):
    data = json.dumps(params)
    with open(file_path, 'w') as f:
        f.write(data)


def load_params(file_path: str):
    with open(file_path) as f:
        data = json.load(f)
    return data


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


def save_autoencoder(encoder: torch.nn.Module, decoder: torch.nn.Module, root_dir: str, epoch: int):
    if not os.path.isdir(root_dir):
        os.makedirs(root_dir)

    encoder_path = os.path.join(root_dir, ENCODER_DIR, ENCODER_NAME.format(f'_ep{epoch}'))
    decoder_path = os.path.join(root_dir, DECODER_DIR, DECODER_NAME.format(f'_ep{epoch}'))

    torch.save(encoder.state_dict(), encoder_path)
    torch.save(decoder.state_dict(), decoder_path)


def load_autoencoder(root_dir: str, epoch: int):
    params, encoder_params, decoder_params = load_autoencoder_params(root_dir)

    encoder = Encoder((params['in_channels'], params['N'], params['block_size']), params['representation_dim'], **encoder_params)
    encoder_path = os.path.join(root_dir, ENCODER_DIR, ENCODER_NAME.format(f'_ep{epoch}'))
    encoder.load_state_dict(torch.load(encoder_path))

    decoder = Decoder(params['representation_dim'], (params['in_channels'], params['N'], params['block_size']), **decoder_params)
    decoder_path = os.path.join(root_dir, DECODER_DIR, DECODER_NAME.format(f'_ep{epoch}'))
    decoder.load_state_dict(torch.load(decoder_path))

    return encoder, decoder


def save_data_list(data: t.List[t.Any], path: str, mode:str = 'a'):
    data_str = [str(x) for x in data]
    with open(path, mode) as f:
        f.write(DELIMITER.join(data_str))


def plot_convergence(results_path: str, save_path: str, read_label: bool = False):
    if read_label:
        skip_rows = 1
        with open(results_path) as f:
            labels = f.readline()
        labels = labels.split(DELIMITER)
    else:
        skip_rows = 0
        labels = None
    data = np.loadtxt(results_path, delimiter= DELIMITER, skiprows= skip_rows)

    if not labels:
        labels = [f'{i}' for i in range(len(data[0, :]))]

    for i in range(1, len(data[0, :])):
        plt.plot(data[:, 0], data[:, i], label = labels[i])
    plt.xlabel(labels[0])
    plt.ylabel('Loss')
    plt.savefig(save_path)
    plt.close()
