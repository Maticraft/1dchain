from src.models.utils import get_edges
from src.models.autoencoder import Decoder, Encoder


import torch
import torch.nn as nn


import typing as t


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