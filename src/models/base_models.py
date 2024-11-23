import typing as t

import torch.nn as nn
import torch



class MLP(nn.Module):
    def __init__(self, layers_num: int, input_size: int, hidden_size: int, output_size: int, activation: str = 'relu', final_activation: str = 'self'):
        super().__init__()
        self.activation = activation
        self.final_activation = final_activation
        self.nn = self._initialize_layers(layers_num, input_size, hidden_size, output_size)
        
    def _initialize_layers(self, layers_num: int, input_size: int, hidden_size: int, output_size: int) -> nn.Sequential:
        layers = []
        for i in range(layers_num):
            if i == 0:
                layers.append(nn.Linear(input_size, hidden_size))
                layers.append(nn.BatchNorm1d(hidden_size))
            elif i == (layers_num - 1):
                layers.append(nn.Linear(hidden_size, output_size))
            else:
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(nn.BatchNorm1d(hidden_size)) 
            layers.append(self._get_activation(self.activation) if i != (layers_num - 1) else self._get_activation(self.final_activation))
        return nn.Sequential(*layers)
    
    def _get_activation(self, activation: str):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU(negative_slope=0.01)
        elif activation == 'sigmoid':
           return nn.Sigmoid()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'self':
            return self._get_activation(self.activation)
        elif activation == 'none':
            return nn.Identity()
        else:
            return ValueError(f'Activation function: {activation} not implemented')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.nn(x)
    

class ResidualBlock(nn.Module):
    def __init__(self, hidden_size: int):
        super(ResidualBlock, self).__init__()
        self.nn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.nn(x)
    

class UNetLikeEncoder(nn.Module):
    def __init__(self, layers_num: int, input_size: int, hidden_size: int, output_size: int):
        super(UNetLikeEncoder, self).__init__()
        self.nn = self._initialize_layers(layers_num, input_size, hidden_size, output_size)
        
    def _initialize_layers(self, layers_num: int, input_size: int, hidden_size: int, output_size: int) -> nn.ModuleList:
        layers = []
        for i in range(layers_num):
            if i == 0:
                layers.append(nn.Linear(input_size, hidden_size))
            elif i == (layers_num - 1):
                layers.append(nn.Linear(hidden_size, output_size))
            else:
                layers.append(ResidualBlock(hidden_size))
        return nn.ModuleList(layers)
    
    def forward(self, x: torch.Tensor) -> t.List[torch.Tensor]:
        '''
        Args:
          x: torch.Tensor with shape (..., input_size)
        Returns:
          List[layers_num x torch.Tensor] with shape (..., output_size)
        '''
        output = []
        for layer in self.nn:
            x = layer(x)
            output.append(x)
        return output
    

class UNetLikeDecoder(nn.Module):
    def __init__(self, layers_num: int, input_size: int, hidden_size: int, output_size: int):
        super(UNetLikeDecoder, self).__init__()
        self.nn = self._initialize_layers(layers_num, input_size, hidden_size, output_size)

    def _initialize_layers(self, layers_num: int, input_size: int, hidden_size: int, output_size: int) -> nn.ModuleList:
        layers = []
        for i in range(layers_num):
            if i == 0:
                layers.append(nn.Linear(input_size, hidden_size))
            elif i == (layers_num - 1):
                layers.append(nn.Linear(hidden_size, output_size))
            else:
                layers.append(ResidualBlock(hidden_size))
        return nn.ModuleList(layers)
    
    def forward(self, input: t.Iterable[torch.Tensor]) -> torch.Tensor:
        x = self.nn[0](input[-1])
        for i, layer in enumerate(self.nn[1:]):
            x = layer(x + input[-(i + 2)])
        return x
