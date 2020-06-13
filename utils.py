from typing import Union, Tuple

import numpy
import torch
from matplotlib import pyplot
from torch import Tensor
from torch.utils import data as _data


def show_spectrogram(spectrogram: Union[numpy.ndarray, Tensor]) -> None:
    pyplot.pcolormesh(spectrogram)
    pyplot.tight_layout()
    pyplot.show()


def calc_mean_std(data_loader: _data.DataLoader) -> Tuple[float, float]:
    running_mean = 0
    running_std = 0

    i = 0
    for samples, _ in data_loader:
        std, mean = torch.std_mean(samples)
        running_mean += mean.item()
        running_std += std.item()
        i += 1

    dataset_mean = running_mean / i
    dataset_std = running_std / i
    return dataset_mean, dataset_std


def calc_pos_weight(data_loader: _data.DataLoader) -> Tensor:
    _, labels = next(iter(data_loader))
    cum_pos = torch.zeros(labels.shape[1], device=labels.device)
    cum_neg = torch.zeros(labels.shape[1], device=labels.device)

    for _, labels in data_loader:
        pos = labels.sum(0)
        neg = torch.ones_like(labels).sum(0) - pos
        cum_pos += pos
        cum_neg += neg

    pos_weight = cum_neg / (cum_pos + 1e-5)
    return pos_weight
