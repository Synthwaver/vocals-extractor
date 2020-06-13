import abc
import random
from typing import Tuple, Sequence, Iterator, Any, Generator
import itertools
import pathlib

import pandas
import torch

from torch import Tensor
from torch.nn import functional as _func
from torch.utils.data import dataset as _data

from audio import Audio


class IterableAudioDataset(_data.IterableDataset):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __iter__(self) -> Iterator: pass

    @abc.abstractmethod
    def __len__(self) -> int: pass

    @staticmethod
    def get_samples_from_waveform(
            waveform: Tensor,
            sample_length: int,
            stft_n_fft: int,
            stft_win_length: int,
            stft_hop: int,
            shuffle: bool = False,
            seed: Any = None
    ) -> Generator:
        amplitude, _ = Audio.compute_stft(
            waveform, stft_n_fft, stft_win_length, stft_hop)
        spectrogram = Audio.get_spectrogram(amplitude, top_db=80)
        return IterableAudioDataset.get_samples_from_spectrogram(
            spectrogram, sample_length, shuffle=shuffle, seed=seed)

    @staticmethod
    def get_samples_from_spectrogram(
            spectrogram: Tensor,
            sample_length: int,
            shuffle: bool = False,
            seed: Any = None
    ) -> Generator:
        torch_image_like = spectrogram.unsqueeze(0)
        half = sample_length // 2
        padding = [half, half, 0, 0]
        padded = _func.pad(torch_image_like, padding, mode='constant', value=0)
        #normalized = IterableAudioDataset.normalize_samples(padded)
        normalized = padded

        original_length = spectrogram.shape[1]
        indices = list(range(original_length))
        if shuffle:
            indices = IterableAudioDataset.shuffle_indices(indices, seed)
        for x in indices:
            yield normalized[:, :, x: x + half * 2 + 1]

    # @staticmethod
    # def normalize_samples(data: Tensor) -> Tensor:
    #     mean = [-17.2000]
    #     std = [13.5918]
    #     normalize = torchvision.transforms.Normalize(mean, std)
    #     return normalize(data)

    @staticmethod
    def shuffle_indices(x: Sequence, seed: Any = None) -> Sequence:
        result = x
        random.Random(seed).shuffle(result)
        return result


class TrainDataset(IterableAudioDataset):

    def __init__(
            self,
            data_frame: pandas.DataFrame,
            audio_sample_rate: int,
            sample_length: int,
            stft_n_fft: int,
            stft_win_length: int,
            stft_hop: int,
            shuffle: bool = False,
            seed: Any = None,
            device: torch.device = torch.device("cpu")
    ) -> None:
        self.data = pandas.DataFrame()
        self.audio_sample_rate: int = audio_sample_rate
        self.sample_length: int = sample_length
        self.stft_n_fft: int = stft_n_fft
        self.stft_win_length: int = stft_win_length
        self.stft_hop: int = stft_hop
        self.shuffle: bool = shuffle
        self.seed: Any = seed if (seed is not None) else random.random()
        self.device = device

        for i, row in data_frame.iterrows():
            new_row = dict()

            for col in ["mixture", "vocals", "accompaniment"]:
                path = pathlib.Path(row[col]).absolute()
                if not path.exists() or path.is_dir():
                    raise FileNotFoundError(path)
                new_row[col] = str(path)

            waveform_length = Audio.calc_waveform_length(
                row["mixture"], self.audio_sample_rate)
            stft_length = Audio.calc_stft_length(
                waveform_length, self.stft_hop)
            new_row["n_frames"] = stft_length

            self.data = self.data.append(new_row, ignore_index=True)

    def __iter__(self) -> Iterator[Tuple[Tensor, Tensor]]:
        indices = list(range(len(self.data)))
        if self.shuffle:
            self.seed = random.Random(self.seed).random()
            indices = IterableAudioDataset.shuffle_indices(indices, self.seed)

        samples = itertools.chain.from_iterable(map(self.get_samples, indices))
        labels = itertools.chain.from_iterable(map(self.get_labels, indices))
        return zip(samples, labels)

    def __len__(self) -> int:
        return int(self.data["n_frames"].sum())

    def get_samples(self, index: int) -> Generator:
        mixture = Audio.load(
            self.data["mixture"][index],
            self.audio_sample_rate,
            device=self.device)
        return IterableAudioDataset.get_samples_from_waveform(
            mixture, self.sample_length,
            self.stft_n_fft, self.stft_win_length, self.stft_hop,
            shuffle=self.shuffle, seed=self.seed)

    def get_labels(self, index: int) -> Generator:
        vocals = Audio.load(
            self.data["vocals"][index],
            self.audio_sample_rate,
            device=self.device)
        accompaniment = Audio.load(
            self.data["accompaniment"][index],
            self.audio_sample_rate,
            device=self.device)
        return TrainDataset.get_labels_from_waveform(
            vocals, accompaniment,
            self.stft_n_fft, self.stft_win_length, self.stft_hop,
            shuffle=self.shuffle, seed=self.seed)

    @staticmethod
    def get_labels_from_waveform(
            vocals: Tensor,
            accompaniment: Tensor,
            stft_n_fft: int,
            stft_win_length: int,
            stft_hop: int,
            shuffle: bool = False,
            seed: Any = None
    ) -> Generator:
        v_amplitude, _ = Audio.compute_stft(
            vocals, stft_n_fft, stft_win_length, stft_hop)
        v_spectrogram = Audio.get_spectrogram(v_amplitude, top_db=60)
        a_amplitude, _ = Audio.compute_stft(
            accompaniment, stft_n_fft, stft_win_length, stft_hop)
        a_spectrogram = Audio.get_spectrogram(a_amplitude, top_db=60)
        return TrainDataset.get_labels_from_spectrogram(
            v_spectrogram, a_spectrogram, shuffle=shuffle, seed=seed)

    @staticmethod
    def get_labels_from_spectrogram(
            vocals: Tensor,
            accompaniment: Tensor,
            shuffle: bool = False,
            seed: Any = None
    ) -> Generator:
        device = vocals.device
        vocals_pos = vocals.where(vocals > 0, torch.tensor(0., device=device))
        accompaniment_pos = accompaniment.where(
            accompaniment > 0, torch.tensor(0., device=device))

        binary_mask = torch.where(
            vocals_pos > accompaniment_pos,
            torch.tensor(1., device=device),
            torch.tensor(0., device=device))

        indices = list(range(binary_mask.shape[1]))
        if shuffle:
            indices = IterableAudioDataset.shuffle_indices(indices, seed)
        for x in indices:
            yield binary_mask[:, x]

    @staticmethod
    def from_csv(
            audio_sample_rate: int,
            sample_length: int,
            stft_n_fft: int,
            stft_win_length: int,
            stft_hop: int,
            csv_path: str,
            delimiter: str = ",",
            shuffle: bool = False,
            seed: Any = None,
            device: torch.device = torch.device("cpu")
    ) -> 'TrainDataset':
        csv_path = pathlib.Path(csv_path)
        if not csv_path.exists() or csv_path.is_dir():
            raise FileNotFoundError(csv_path)

        csv_dir = csv_path.absolute().parent
        df = pandas.read_csv(csv_path, delimiter=delimiter)
        processed_df = pandas.DataFrame()

        for i, row in df.iterrows():
            new_row = dict()
            for col in ["mixture", "vocals", "accompaniment"]:
                path = pathlib.Path(row[col])
                if not path.is_absolute():
                    path = pathlib.Path(csv_dir, path)
                new_row[col] = str(path)
            processed_df = processed_df.append(new_row, ignore_index=True)

        return TrainDataset(
            processed_df, audio_sample_rate, sample_length,
            stft_n_fft, stft_win_length, stft_hop,
            shuffle=shuffle, seed=seed, device=device)


class TestDataset(IterableAudioDataset):

    def __init__(
            self,
            mixture_path: str,
            audio_sample_rate: int,
            sample_length: int,
            stft_n_fft: int,
            stft_win_length: int,
            stft_hop: int,
            device: torch.device = torch.device("cpu")
    ) -> None:
        self.sample_length: int = sample_length
        self.audio_sample_rate: int = audio_sample_rate
        self.stft_n_fft: int = stft_n_fft
        self.stft_win_length = stft_win_length
        self.stft_hop: int = stft_hop

        waveform_length = Audio.calc_waveform_length(
            mixture_path, audio_sample_rate)
        self.samples_count = Audio.calc_stft_length(waveform_length, stft_hop)

        self.waveform = Audio.load(
            mixture_path, sample_rate=audio_sample_rate, device=device)
        self.amplitude, self.phase = Audio.compute_stft(
            self.waveform, self.stft_n_fft,
            self.stft_win_length, self.stft_hop)

    def __iter__(self) -> Iterator[Tensor]:
        spectrogram = Audio.get_spectrogram(self.amplitude, top_db=80)
        return IterableAudioDataset.get_samples_from_spectrogram(
            spectrogram, self.sample_length)

    def __len__(self) -> int:
        return self.amplitude.shape[1]
