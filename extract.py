import argparse

from torch.utils import data as _data

from audio import Audio
from config import Config
from dataset import TestDataset
from model import VocalExtractor


def extract(mixture_path: str, result_path: str):
    vocal_extractor = VocalExtractor(
        Config.stft_n_fft // 2 + 1,
        Config.sample_length,
        pretrained_model=Config.pretrained_model,
        device=Config.device)

    test_ds = TestDataset(
        mixture_path,
        Config.audio_sample_rate,
        Config.sample_length,
        Config.stft_n_fft,
        Config.stft_win_length,
        Config.stft_hop,
        device=Config.device)

    test_loader = _data.DataLoader(test_ds, batch_size=Config.batch_size)

    mask = vocal_extractor.predict(test_loader)
    result_amplitude = Audio.apply_mask(test_ds.amplitude, mask)
    result_waveform = Audio.compute_istft(
        result_amplitude, test_ds.phase, Config.stft_n_fft, Config.stft_hop)

    Audio.save(result_waveform, result_path, Config.audio_sample_rate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--output', required=True)
    namespace = parser.parse_args()

    print("Script started")
    extract(namespace.input, namespace.output)
    print("Extracted successfully")
