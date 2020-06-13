import argparse

from torch.utils import data as _data

from config import Config
from dataset import TrainDataset
from model import VocalExtractor


def fit(train_csv: str, val_csv: str):
    vocal_extractor = VocalExtractor(
        Config.stft_n_fft // 2 + 1,
        Config.sample_length,
        pretrained_model=Config.pretrained_model,
        device=Config.device)

    train_ds = TrainDataset.from_csv(
        Config.audio_sample_rate,
        Config.sample_length,
        Config.stft_n_fft,
        Config.stft_win_length,
        Config.stft_hop,
        train_csv,
        shuffle=True,
        device=Config.device)

    val_ds = TrainDataset.from_csv(
        Config.audio_sample_rate,
        Config.sample_length,
        Config.stft_n_fft,
        Config.stft_win_length,
        Config.stft_hop,
        val_csv,
        shuffle=False,
        device=Config.device)

    train_loader = _data.DataLoader(
        train_ds, batch_size=Config.batch_size, drop_last=True)
    val_loader = _data.DataLoader(val_ds, batch_size=Config.batch_size)

    vocal_extractor.fit(
        train_loader, val_loader,
        epochs=Config.epochs,
        batches=Config.batches,
        learning_rate=Config.learning_rate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', required=True)
    parser.add_argument('-v', '--validation', required=True)
    namespace = parser.parse_args()

    print("Script started")
    fit(namespace.train, namespace.validation)
    print("Training complete")
