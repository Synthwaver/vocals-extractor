import math
from typing import Tuple

import torch
import torchaudio as ta
from torch import Tensor
from torchaudio import transforms as _transf
from torchaudio import functional as _func


class Audio:

    @staticmethod
    def load(
            path: str,
            sample_rate: int,
            mono: bool = True,
            device: torch.device = torch.device("cpu")
    ) -> Tensor:
        waveform, original_sr = ta.load(path)
        waveform = waveform.to(device)
        if original_sr != sample_rate:
            resample = _transf.Resample(original_sr, sample_rate).to(device)
            waveform = resample(waveform)
        if mono:
            channels_dim = 0
            channels_count = waveform.shape[channels_dim]
            waveform = waveform.sum(dim=channels_dim) / channels_count
        return waveform

    @staticmethod
    def save(waveform: Tensor, path: str, sample_rate: int) -> None:
        ta.save(path, waveform.cpu(), sample_rate)

    @staticmethod
    def compute_stft(
            waveform: Tensor,
            n_fft: int,
            win_length: int,
            hop_length: int
    ) -> Tuple[Tensor, Tensor]:
        device = waveform.device
        spectrogram = _transf.Spectrogram(
            n_fft=n_fft, win_length=win_length, hop_length=hop_length,
            power=None).to(device)
        amplitude, phase = _func.magphase(spectrogram(waveform))
        return amplitude, phase

    @staticmethod
    def compute_istft(
            amplitude: Tensor,
            phase: Tensor,
            n_fft: int,
            hop_length: int
    ) -> Tensor:
        real = amplitude * torch.cos(phase)
        imag = amplitude * torch.sin(phase)
        stft = torch.stack((real, imag), dim=-1)
        return _func.istft(stft, n_fft=n_fft, hop_length=hop_length)

    @staticmethod
    def get_spectrogram(amplitude: Tensor, top_db: float = 80) -> Tensor:
        device = amplitude.device
        amplitude_to_db = _transf.AmplitudeToDB(top_db=top_db).to(device)
        return amplitude_to_db(amplitude.pow(2))

    @staticmethod
    def apply_mask(data: Tensor, mask: Tensor) -> Tensor:
        thresholded_mask = mask.where(
            mask > 0.15, torch.tensor(0., device=mask.device))
        return data * thresholded_mask

    @staticmethod
    def calc_waveform_length(path: str, sample_rate: int) -> int:
        info, _ = ta.info(path)
        return math.ceil(info.length * sample_rate / info.rate / info.channels)

    @staticmethod
    def calc_stft_length(waveform_length: int, hop_length: int) -> int:
        return (waveform_length + hop_length) // hop_length
