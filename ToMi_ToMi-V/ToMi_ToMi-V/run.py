#!/usr/bin/env python3
"""Train or evaluate ToMi and ToMi-V."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_provider.series import DATASETS, read_series
from engine.trainer import evaluate, fit, load_checkpoint, write_json
from models import MODELS, ModelConfig


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--data', choices=DATASETS, required=True)
    parser.add_argument('--root_path', type=Path, default=Path('.'))
    parser.add_argument('--data_path', type=Path, required=True)
    parser.add_argument('--features', choices=('M', 'MS', 'S'), default='M')
    parser.add_argument('--target', default=None)
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--window_size', type=int, required=True)
    parser.add_argument('--enc_in', type=int, default=None, help='Inferred from data unless supplied')
    parser.add_argument('--embed_size', type=int, default=16)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--num_hidden_layers', type=int, default=1)
    parser.add_argument('--d_ff', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--local_kernel_size', type=int, default=5)
    parser.add_argument('--channel_mixer_rank', type=int, default=4)
    parser.add_argument('--channel_mixer_dropout', type=float, default=None)
    parser.add_argument('--channel_mixer_gate_init', type=float, default=0.0)
    parser.add_argument('--alpha', type=float, default=0.3)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=1)
    parser.add_argument('--train_epochs', type=int, default=60)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--lradj', choices=('type1', 'constant'), default='type1')
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_threads', type=int, default=2)
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda:0, ...')
    parser.add_argument('--output_dir', type=Path, default=Path('outputs'))
    parser.add_argument('--mode', choices=('train', 'test'), default='train')
    parser.add_argument('--checkpoint', type=Path, default=None)
    parser.add_argument('--inverse', action='store_true', help='Report metrics in the original dataset units')
    parser.add_argument('--skip_completed', action='store_true')
    return parser


def main():
    args = build_parser().parse_args()
    for name in ('seq_len', 'pred_len', 'batch_size', 'eval_batch_size', 'train_epochs', 'patience', 'num_threads'):
        if getattr(args, name) < 1:
            raise ValueError(f'{name} must be positive')
    if args.num_workers < 0 or args.learning_rate <= 0 or not 0 <= args.alpha <= 1:
        raise ValueError('Check num_workers, learning_rate, and alpha')
    if args.mode == 'test' and args.checkpoint is None:
        raise ValueError('--mode test requires --checkpoint')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.num_threads)
    device = torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu')
                          if args.device == 'auto' else args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available')
    path = args.root_path / args.data_path
    series = read_series(path, args.data, args.seq_len, args.pred_len, args.features,
                         args.target, args.train_ratio, args.test_ratio)
    channels = series.values.shape[1]
    if args.enc_in is not None and args.enc_in != channels:
        raise ValueError(f'--enc_in={args.enc_in} differs from the dataset channels ({channels})')
    fields = ModelConfig.__dataclass_fields__
    config = ModelConfig(**{k: channels if k == 'enc_in' else getattr(args, k) for k in fields})
    file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    identity = {'model': args.model, 'config': asdict(config), 'data': args.data,
                'data_sha256': file_hash, 'features': args.features, 'target': args.target,
                'alpha': args.alpha, 'learning_rate': args.learning_rate, 'batch_size': args.batch_size,
                'train_epochs': args.train_epochs, 'patience': args.patience, 'lradj': args.lradj,
                'seed': args.seed, 'train_ratio': args.train_ratio, 'test_ratio': args.test_ratio,
                'inverse_metrics': args.inverse, 'mode': args.mode,
                'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
                if args.mode == 'test' else None}
    run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    directory = args.output_dir / f'{args.model}_{args.data}_L{args.seq_len}_H{args.pred_len}_w{args.window_size}_{run_id}'
    directory.mkdir(parents=True, exist_ok=True)
    if args.skip_completed and (directory / 'metrics.json').is_file():
        print(f'Skip completed: {directory}', flush=True)
        return
    # One process owns a run directory; release automatically on process exit.
    import fcntl
    with (directory / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.skip_completed and (directory / 'metrics.json').is_file():
            print(f'Skip completed: {directory}', flush=True)
            return
        if (directory / 'metrics.json').exists() or (directory / 'checkpoint.pth').exists():
            raise FileExistsError('Run artifacts already exist; use --skip_completed or a new --output_dir')
        write_json(directory / 'config.json', {'run_id': run_id, **identity,
                   'dataset_columns': series.columns, 'boundaries': series.boundaries,
                   'device': str(device), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                   'validation_policy': 'All validation windows; chronological; sample-weighted MSE',
                   'torch_version': str(torch.__version__)})
        model = MODELS[args.model](config).float().to(device)
        print(f'{args.model} device={device} input=(*,{args.seq_len},{channels}) '
              f'output=(*,{args.pred_len},{channels}) w={args.window_size} output_dir={directory}', flush=True)
        training = fit(model, series, args, config, directory, device) if args.mode == 'train' else {}
        checkpoint_path = directory / 'checkpoint.pth' if args.mode == 'train' else args.checkpoint
        checkpoint = load_checkpoint(model, checkpoint_path, device, config, series.columns, args.model)
        if not (np.array_equal(checkpoint['scaler_mean'], series.scaler.mean_)
                and np.array_equal(checkpoint['scaler_scale'], series.scaler.scale_)):
            raise ValueError('Dataset training statistics differ from the saved checkpoint')
        test_loader = DataLoader(series.dataset('test'), batch_size=args.eval_batch_size,
                                 shuffle=False, num_workers=args.num_workers)
        metrics = evaluate(model, test_loader, device, series.target_index,
                           series.scaler if args.inverse else None)
        metrics.update(run_id=run_id, model=args.model, dataset=args.data, seq_len=args.seq_len,
                       pred_len=args.pred_len, window_size=args.window_size, alpha=args.alpha,
                       seed=args.seed, best_epoch=checkpoint['epoch'],
                       metric_scale='original' if args.inverse else 'standardized', **training)
        write_json(directory / 'metrics.json', metrics)
        print(json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    main()
