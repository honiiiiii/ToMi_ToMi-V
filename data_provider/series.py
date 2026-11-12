"""Chronological forecasting splits with train-only standardization."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


DATASETS = ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'ECL', 'Traffic',
            'Weather', 'Solar', 'Exchange', 'custom')


@dataclass
class SeriesData:
    values: np.ndarray
    columns: list[str]
    scaler: StandardScaler
    boundaries: dict[str, tuple[int, int]]
    seq_len: int
    pred_len: int
    target_index: int | None

    def dataset(self, split):
        start, stop = self.boundaries[split]
        return ForecastWindows(self.values[start:stop], self.seq_len, self.pred_len)


class ForecastWindows(Dataset):
    def __init__(self, values, seq_len, pred_len):
        self.values = values
        self.seq_len = seq_len
        self.pred_len = pred_len
        if len(self) < 1:
            raise ValueError('Split is too short for the requested input and prediction lengths')

    def __len__(self):
        return len(self.values) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        end = index + self.seq_len
        x = self.values[index:end].astype(np.float32)
        y = self.values[end:end + self.pred_len].astype(np.float32)
        return x, y


def read_series(path, dataset, seq_len, pred_len, features='M', target=None,
                train_ratio=0.7, test_ratio=0.2):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Dataset file not found: {path}')
    if dataset == 'Solar' and path.suffix.lower() in {'.txt', '.csv'}:
        values = np.loadtxt(path, delimiter=',', dtype=np.float64, ndmin=2)
        columns = [str(i) for i in range(values.shape[1])]
    else:
        frame = pd.read_csv(path)
        frame = frame.drop(columns=['date'], errors='ignore')
        if target is not None:
            if target not in frame.columns:
                raise ValueError(f'Target {target!r} is not present in the dataset')
            frame = frame[[c for c in frame.columns if c != target] + [target]]
        columns = list(map(str, frame.columns))
        values = frame.to_numpy(dtype=np.float64)
    if values.ndim != 2 or values.shape[1] == 0 or not np.isfinite(values).all():
        raise ValueError('Dataset must contain a finite numeric [time, variables] array')
    if features in {'S', 'MS'}:
        if target is None or target not in columns:
            raise ValueError('S/MS forecasting requires a valid --target column')
        target_index = columns.index(target)
    else:
        target_index = None
    if features == 'S':
        values = values[:, [target_index]]
        columns, target_index = [target], 0
    length = len(values)
    if dataset.startswith('ETTh') or dataset.startswith('ETTm'):
        rate = 4 if dataset.startswith('ETTm') else 1
        train_end = 12 * 30 * 24 * rate
        val_end = train_end + 4 * 30 * 24 * rate
        test_end = val_end + 4 * 30 * 24 * rate
        if length < test_end:
            raise ValueError(f'{dataset} requires at least {test_end} rows for its standard splits')
    else:
        if not (0 < train_ratio < 1 and 0 < test_ratio < 1 and train_ratio + test_ratio < 1):
            raise ValueError('train_ratio and test_ratio must be positive and sum to less than one')
        train_end = int(length * train_ratio)
        test_size = int(length * test_ratio)
        # Solar follows its conventional separately rounded validation size.
        val_end = (train_end + int(length * (1 - train_ratio - test_ratio) + 1e-9)
                   if dataset == 'Solar' else length - test_size)
        test_end = length
    if train_end < seq_len + pred_len or val_end - train_end < pred_len or test_end - val_end < pred_len:
        raise ValueError('Data splits are too short for seq_len and pred_len')
    scaler = StandardScaler().fit(values[:train_end])
    scaled = scaler.transform(values)
    boundaries = {'train': (0, train_end),
                  'val': (train_end - seq_len, val_end),
                  'test': (test_end - int(length * test_ratio) - seq_len, test_end)
                          if dataset == 'Solar' else (val_end - seq_len, test_end)}
    return SeriesData(scaled, columns, scaler, boundaries, seq_len, pred_len, target_index)
