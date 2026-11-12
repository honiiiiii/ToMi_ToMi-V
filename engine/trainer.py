"""Training and streaming evaluation for both forecasting models."""
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .loss import forecasting_loss


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def select_features(prediction, target, target_index):
    if target_index is not None:
        return prediction[..., target_index:target_index + 1], target[..., target_index:target_index + 1]
    return prediction, target


@torch.no_grad()
def evaluate(model, loader, device, target_index=None, scaler=None):
    model.eval()
    absolute, squared, count = 0.0, 0.0, 0
    for x, y in loader:
        prediction = model(x.to(device))
        prediction, y = select_features(prediction.cpu(), y, target_index)
        if scaler is not None:
            indices = slice(None) if target_index is None else slice(target_index, target_index + 1)
            scale = torch.as_tensor(scaler.scale_[indices], dtype=prediction.dtype)
            mean = torch.as_tensor(scaler.mean_[indices], dtype=prediction.dtype)
            prediction = prediction * scale + mean
            y = y * scale + mean
        error = (prediction - y).double()
        if not torch.isfinite(error).all():
            raise FloatingPointError('Non-finite predictions during evaluation')
        absolute += error.abs().sum().item()
        squared += error.square().sum().item()
        count += error.numel()
    if count == 0:
        raise ValueError('Evaluation loader is empty')
    return {'mse': squared / count, 'mae': absolute / count,
            'rmse': math.sqrt(squared / count), 'evaluated_values': count}


def fit(model, series, args, config, directory, device):
    train_loader = DataLoader(series.dataset('train'), batch_size=args.batch_size,
                              shuffle=True, drop_last=True, num_workers=args.num_workers)
    # Evaluate all validation windows, with a fixed chronological ordering.
    val_loader = DataLoader(series.dataset('val'), batch_size=args.batch_size,
                            shuffle=False, drop_last=False, num_workers=args.num_workers)
    if len(train_loader) == 0:
        raise ValueError('Training split is smaller than batch_size; lower batch_size')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_mse, stale = math.inf, 0
    started = time.monotonic()
    for epoch in range(1, args.train_epochs + 1):
        model.train()
        losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            prediction, target = select_features(model(x), y, series.target_index)
            loss = forecasting_loss(prediction, target, args.alpha)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training loss')
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        validation = evaluate(model, val_loader, device, series.target_index)
        record = {'epoch': epoch, 'train_loss': float(np.mean(losses)),
                  'validation_mse': validation['mse'], 'learning_rate': optimizer.param_groups[0]['lr']}
        with (directory / 'history.jsonl').open('a') as handle:
            handle.write(json.dumps(record) + '\n')
        print(f'Epoch {epoch}/{args.train_epochs} train={record["train_loss"]:.6f} '
              f'val_mse={validation["mse"]:.6f}', flush=True)
        if validation['mse'] <= best_mse:
            best_mse, stale = validation['mse'], 0
            torch.save({'state_dict': model.state_dict(), 'model': args.model,
                        'config': asdict(config), 'epoch': epoch,
                        'validation_mse': best_mse, 'columns': series.columns,
                        'scaler_mean': series.scaler.mean_.tolist(),
                        'scaler_scale': series.scaler.scale_.tolist()}, directory / 'checkpoint.pth')
        else:
            stale += 1
            if stale >= args.patience:
                print('Early stopping', flush=True)
                break
        if args.lradj == 'type1':
            for group in optimizer.param_groups:
                group['lr'] = args.learning_rate * (0.5 ** (epoch - 1))
    return {'training_seconds': time.monotonic() - started, 'best_validation_mse': best_mse}


def load_checkpoint(model, path, device, config, columns, model_name):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint['config'] != asdict(config) or checkpoint['model'] != model_name:
        raise ValueError('Checkpoint model/configuration differs from this command')
    if checkpoint['columns'] != columns:
        raise ValueError('Dataset column order differs from the saved checkpoint')
    model.load_state_dict(checkpoint['state_dict'])
    return checkpoint
