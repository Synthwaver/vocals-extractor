from typing import Optional


class Config:
    # data
    audio_sample_rate: int = 22050
    stft_n_fft: int = 1024
    stft_win_length: int = 1024
    stft_hop: int = 256
    sample_length: int = 9

    # model
    epochs: int = 100
    batches: Optional[int] = None
    batch_size: int = 100
    learning_rate: float = 1e-3
    pretrained_model: Optional[str] = r"models/pretrained.pt"

    # other
    device = "cuda"
