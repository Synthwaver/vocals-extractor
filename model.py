import math
import pathlib
from typing import Dict

import torch
import torchsummary
from ignite import engine as _engine
from ignite import metrics as _metrics
from ignite import handlers as _handlers
from ignite.contrib import handlers as _chandlers
from ignite.contrib.handlers import tensorboard_logger as _tb_logger
from torch import Tensor, nn
from torch import optim as _optim
from torch.utils import data as _data


class Model(nn.Module):

    def __init__(self, freq_bins: int, time_bins: int) -> None:
        super(Model, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=3),
            nn.Conv2d(16, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=3)
        )
        self.classifier = nn.Sequential(
            nn.modules.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(((freq_bins // 9) * (time_bins // 9) * 32), 128),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, freq_bins),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class VocalExtractor:

    def __init__(
            self,
            sample_freq_bins: int,
            sample_time_bins: int,
            pretrained_model: str = None,
            device: torch.device = torch.device("cpu")
    ) -> None:
        self.freq_bins = sample_freq_bins
        self.time_bins = sample_time_bins
        self.model = Model(sample_freq_bins, sample_time_bins).to(device)
        self.device = device

        if pretrained_model is not None:
            file = pathlib.Path(pretrained_model)
            if not file.exists():
                raise FileNotFoundError(file)
            self.model.load_state_dict(torch.load(file))

    def fit(
            self,
            train_loader: _data.DataLoader,
            val_loader: _data.DataLoader,
            epochs: int = 1,
            batches: int = None,
            learning_rate: float = 1e-3
    ) -> None:
        if batches is None:
            batches = VocalExtractor.get_number_of_batches(train_loader)

        loss_fn = nn.BCELoss()
        optimizer = _optim.Adam(
            self.model.parameters(), lr=learning_rate)

        trainer = _engine.create_supervised_trainer(
            self.model, optimizer, loss_fn, device=self.device)

        _metrics.RunningAverage(
            output_transform=lambda x: x, device=self.device
        ).attach(trainer, 'loss')
        progressbar = _chandlers.ProgressBar(
            bar_format=
            "{desc}[{n_fmt}/{total_fmt}] {percentage:3.0f}%|{bar:20}| "
            "[{elapsed}<{remaining}]{postfix}",
            persist=True, ascii=" #"
        )
        progressbar.attach(trainer, ['loss'])

        def get_metrics_fn() -> Dict[str, _metrics.Metric]:
            def rounded_transform(output):
                y_pred, y = output
                return torch.round(y_pred), y

            transform = rounded_transform
            accuracy = _metrics.Accuracy(transform, device=self.device)
            precision = _metrics.Precision(transform, device=self.device)
            recall = _metrics.Recall(transform, device=self.device)
            f1 = precision * recall * 2 / (precision + recall + 1e-20)
            return {
                'loss': _metrics.Loss(loss_fn),
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }

        evaluator = _engine.create_supervised_evaluator(
            self.model, metrics=get_metrics_fn(), device=self.device)

        score_fn_name = "f1"

        def score_function(engine: _engine.Engine):
            return engine.state.metrics[score_fn_name]

        best_model_saver = _handlers.ModelCheckpoint(
            dirname="best_models",
            filename_prefix="vocal_extractor",
            score_name=score_fn_name,
            score_function=score_function,
            n_saved=5,
            create_dir=True
        )
        evaluator.add_event_handler(
            _engine.Events.COMPLETED,
            best_model_saver,
            {"model": self.model})

        each_model_saver = _handlers.ModelCheckpoint(
            dirname="all_models",
            filename_prefix="vocal_extractor",
            score_name=score_fn_name,
            score_function=score_function,
            n_saved=None,
            create_dir=True
        )
        evaluator.add_event_handler(
            _engine.Events.COMPLETED,
            each_model_saver,
            {"model": self.model})

        @trainer.on(_engine.Events.EPOCH_COMPLETED)
        def on_epoch_completed(engine: _engine.Engine) -> None:
            metrics = VocalExtractor.compute_metrics(val_loader, evaluator)
            string = ", ".join(f"val_{k}: {v:.4f}" for k, v in metrics.items())
            progressbar.log_message(string + "\n")

        with _tb_logger.TensorboardLogger(log_dir="tb_logs") as tb_logger:
            global_step = _tb_logger.global_step_from_engine(trainer)

            train_running_loss_log_handler = _tb_logger.OutputHandler(
                tag="training", output_transform=lambda x: {'running_loss': x})
            tb_logger.attach(
                trainer,
                log_handler=train_running_loss_log_handler,
                event_name=_engine.Events.ITERATION_COMPLETED)

            val_metrics_log_handler = _tb_logger.OutputHandler(
                tag="validation",
                metric_names=[name for name, _ in get_metrics_fn().items()],
                global_step_transform=global_step)
            tb_logger.attach(
                evaluator,
                log_handler=val_metrics_log_handler,
                event_name=_engine.Events.EPOCH_COMPLETED)

            tb_logger.attach(
                trainer,
                log_handler=_tb_logger.OptimizerParamsHandler(optimizer),
                event_name=_engine.Events.ITERATION_STARTED)

            tb_logger.attach(
                trainer,
                log_handler=_tb_logger.WeightsScalarHandler(self.model),
                event_name=_engine.Events.ITERATION_COMPLETED)
            tb_logger.attach(
                trainer,
                log_handler=_tb_logger.WeightsHistHandler(self.model),
                event_name=_engine.Events.EPOCH_COMPLETED)

            tb_logger.attach(
                trainer,
                log_handler=_tb_logger.GradsScalarHandler(self.model),
                event_name=_engine.Events.ITERATION_COMPLETED)
            tb_logger.attach(
                trainer,
                log_handler=_tb_logger.GradsHistHandler(self.model),
                event_name=_engine.Events.EPOCH_COMPLETED)

        torchsummary.summary(
            self.model, input_size=(1, self.freq_bins, self.time_bins),
            batch_size=train_loader.batch_size, device=self.device)
        trainer.run(data=train_loader, epoch_length=batches, max_epochs=epochs)

    def predict(self, loader: _data.DataLoader) -> Tensor:
        def estimation_update(engine: _engine.Engine, batch) -> dict:
            return {"y_pred": self.model(batch)}

        estimator = _engine.Engine(estimation_update)
        result = []

        @estimator.on(_engine.Events.ITERATION_COMPLETED)
        def save_results(engine: _engine.Engine) -> None:
            output = engine.state.output['y_pred'].detach()
            result.append(output)
            torch.cuda.empty_cache()

        self.model.eval()
        batches = VocalExtractor.get_number_of_batches(loader)
        estimator.run(loader, epoch_length=batches, max_epochs=1)

        result = torch.cat(result, dim=0)
        return result.transpose(0, 1)

    @staticmethod
    def get_number_of_batches(data_loader: _data.DataLoader) -> int:
        batches = len(data_loader) / data_loader.batch_size
        if data_loader.drop_last:
            return math.floor(batches)
        else:
            return math.ceil(batches)

    @staticmethod
    def compute_metrics(
            loader: _data.DataLoader,
            evaluator: _engine.Engine
    ) -> Dict[str, float]:
        batches = VocalExtractor.get_number_of_batches(loader)
        return evaluator.run(loader, epoch_length=batches).metrics
