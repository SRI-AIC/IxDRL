import logging
import random
from typing import Union

import numpy as np
import torch.nn as nn

import torch

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'


def init_seed(seed: int = 0) -> np.random.RandomState:
    """
    Initializes the random number generators for PyTorch, Numpy and native random modules.
    :param int seed: the seed with which to initialize the generators.
    :rtype: np.random.RandomState
    :return: a Numpy's random number generator initialized with the given seed.
    """
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return random_state


def select_device(cuda: bool, log: bool = True) -> torch.device:
    """
    Selects a PyTorch device with the given specification.
    :param bool cuda: whether to get the CUDA device, if available.
    :param bool log: whether to log the selected device.
    :rtype: torch.device
    :return: the selected PyTorch device object.
    """
    if cuda and torch.cuda.is_available():
        torch.set_default_device('cuda')
        device = torch.device(torch.cuda.current_device())
    else:
        device = torch.device('cpu')
    if log:
        logging.info(f'Selected device: {device}')
    return device


def get_num_params(model: nn.Module, trainable: bool = True) -> int:
    """
    Counts the number of (trainable) parameters in a given model.
    See: https://stackoverflow.com/a/49201237/16031961
    See: https://stackoverflow.com/a/62764464/16031961
    :param nn.Module model: the model.
    :param bool trainable: include only trainable parameters in the count.
    :rtype: int
    :return: the number of (trainable) parameters in the model.
    """
    return sum(dict((p.data_ptr(), p.numel())
                    for p in model.parameters()
                    if not trainable or p.requires_grad).values())


def rnn_weights_init(m: torch.nn, generator: torch.Generator, mean: float = 0.0, std: float = 0.02):
    # custom weights initialization
    for c in m.children():
        classname = c.__class__.__name__
        if classname.find('GRU') != -1:
            for k, v in c.named_parameters():
                if 'weight' in k:
                    v.data.normal_(mean, std, generator=generator)


class MLP(nn.Module):
    """
    A simple multi-layer perceptron (MLP) model with fixed hidden layer size.
    """

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 hidden_size: int,
                 num_layers: int = 1,
                 dropout: float = 0):
        super().__init__()
        self.input_size = input_size

        # hidden layers
        self.model = nn.Sequential()
        for _ in range(num_layers):
            self.model.append(nn.Linear(input_size, hidden_size))
            self.model.append(nn.ReLU(inplace=True))
            self.model.append(nn.Dropout(dropout))
            input_size = hidden_size

        # output
        self.model.append(nn.Linear(hidden_size, output_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, input_size)
        # check dims
        assert len(x.shape) == 2, \
            f'Expected input of shape (batch, input_size), but got: {x.shape}'
        assert x.shape[-1] == self.input_size, \
            f'Input size: {x.shape[-1]} does not match expected size: {self.input_size}'

        return self.model(x)  # shape: (batch, output_size


class EarlyStopper:
    """
    A method to interrupt training (early stop) when the validation score does not increase for a certain number of
    trials/epochs.
    from: https://stackoverflow.com/a/73704579/16031961
    """

    def __init__(self, max_steps: int, min_delta: float = 0):
        self.max_steps = max_steps
        self.min_delta = min_delta
        self.num_steps = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss: Union[torch.Tensor, float]) -> bool:
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.num_steps = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.num_steps += 1
            if self.num_steps >= self.max_steps:
                return True
        return False
