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
        else:
            return ValueError(f'Activation function: {activation} not implemented')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.nn(x)
    