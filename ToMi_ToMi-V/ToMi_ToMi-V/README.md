# ToMi and ToMi-V

Anonymous source release for multivariate time series forecasting. The two
models share a temporal token-mixing encoder and forecasting head. `tomi`
processes variables independently; `tomi-v` adds gated low-rank
cross-variable mixing after temporal encoding.

The archive contains source code, shell scripts, dependency specifications,
tests, and documentation. Datasets, checkpoints, experiment records, and
other forecasting models are not included.

## Installation

Use Python 3.10 or newer on Linux. Create an environment and install the
tested dependency versions from `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For CUDA execution, install a PyTorch build compatible with the local GPU
and driver. CPU execution is supported through `--device cpu`.

## Data preparation

Place datasets under a directory of your choice and pass their paths to
`--data_path`. CSV files should have a header, an optional `date` column,
and numeric variable columns. Solar accepts its headerless comma-separated
numeric text file. Missing or non-finite values must be handled before
running. No data is downloaded automatically.

| `--data` | Input format | Chronological train / validation / test split |
| --- | --- | --- |
| `ETTh1`, `ETTh2` | CSV | 12 / 4 / 4 months, using 30-day months |
| `ETTm1`, `ETTm2` | CSV | Same split at four samples per hour |
| `ECL`, `Traffic`, `Weather`, `Exchange`, `custom` | CSV | 70% / 10% / 20% by default |
| `Solar` | Headerless comma-separated text | 70% / 10% / 20%, rounded as separate sizes |

For non-ETT data, `--train_ratio` and `--test_ratio` allow different split
ratios. Standardization is fitted only on the training partition. The
validation and test partitions include the preceding input context;
their predicted targets remain in their own partitions. Date features
and decoder inputs are unnecessary for these models.

## Run a forecast

`--window_size` is required. It controls temporal region length and must
be between 1 and `--seq_len`. Input length and prediction horizon are
separate parameters. The commands below illustrate usage; they are not
claims about dataset-specific optimal hyperparameters.

```bash
# ToMi
CUDA_VISIBLE_DEVICES=0 python -u run.py \
  --model tomi --data ETTh1 --data_path dataset/ETTh1.csv \
  --seq_len 96 --pred_len 96 --window_size 24 \
  --d_model 128 --n_heads 8 --num_hidden_layers 1 \
  --learning_rate 0.001 --dropout 0.1 --alpha 0.3 \
  --batch_size 32 --train_epochs 60 --device cuda:0

# ToMi-V
CUDA_VISIBLE_DEVICES=0 python -u run.py \
  --model tomi-v --data ETTh1 --data_path dataset/ETTh1.csv \
  --seq_len 96 --pred_len 96 --window_size 24 \
  --d_model 128 --n_heads 8 --num_hidden_layers 1 \
  --channel_mixer_rank 4 --channel_mixer_dropout 0.1 \
  --channel_mixer_gate_init 0 --alpha 0.3 \
  --train_epochs 10 --device cuda:0
```

The number of variables is inferred from the file. An optional `--enc_in`
checks it explicitly. The default `--features M` predicts all variables.
`--features S --target <column>` uses one series; `--features MS --target
<column>` uses all input variables but supervises and evaluates only the
target. Passing `--target` places that column last consistently.

Use `python run.py --help` for every option. Parameter definitions are in
[docs/CONFIGURATION.md](docs/CONFIGURATION.md).

## Run all four prediction horizons

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_forecasting.sh \
  tomi ETTh1 dataset/ETTh1.csv 24 \
  --seq_len 96 --d_model 128 --n_heads 8 --device cuda:0

CUDA_VISIBLE_DEVICES=0 bash scripts/run_forecasting.sh \
  tomi-v ETTh1 dataset/ETTh1.csv 24 \
  --seq_len 96 --channel_mixer_rank 4 --train_epochs 10 --device cuda:0
```

The positional arguments are model, dataset, data path, and window size.
The script evaluates horizons `96 192 336 720` sequentially. Override them
using `PRED_LENGTHS="96 192"`. Additional arguments are forwarded to
`run.py`. Keep `--pred_len` out of the forwarded arguments so the loop
can set it. Relative data paths are interpreted from the working
directory where you invoke the script.

Completed identical runs are skipped. A failed run, including CUDA OOM,
is logged and the script continues to the next horizon. It does not
silently reduce batch size or modify the model. If any horizon fails,
the script returns a nonzero status after finishing the queue.

For detached execution:

```bash
mkdir -p logs
nohup env CUDA_VISIBLE_DEVICES=0 bash scripts/run_forecasting.sh \
  tomi-v ETTh1 dataset/ETTh1.csv 24 --train_epochs 10 --device cuda:0 \
  > logs/queue.log 2>&1 < /dev/null &
```

## Training, metrics, and checkpoints

Training uses Adam and the joint loss

```text
(1 - alpha) * (MAE + MSE)
    + alpha * mean(abs(rFFT(prediction) - rFFT(target)))
```

The Fourier transform is applied along the prediction horizon with its
unnormalized forward convention. No Fourier transform is needed during
inference. The engine trains in float32 without mixed precision. With
`--lradj type1`, learning rates follow the epoch-wise halving schedule;
`--lradj constant` disables this schedule.

Checkpoints are selected only by validation MSE, with patience-based
early stopping. This standalone engine evaluates **all validation
windows in chronological order** and computes sample-weighted MSE.
Training shuffles windows and drops the final incomplete training batch.
The test partition is evaluated after loading the selected checkpoint.
Seeds are set, but numerical results can still depend on the software
and hardware environment.

Each run creates a configuration-specific directory in `outputs/`:

```text
config.json       model, data checksum, split boundaries, and training settings
history.jsonl     per-epoch training loss and validation MSE
checkpoint.pth    best weights, model configuration, and normalization statistics
metrics.json      paired test MSE / MAE and run metadata
```

Metrics use the training-standardized dataset scale by default; the
model's inverse RevIN is always applied. `--inverse` additionally
reverses dataset standardization when reporting metrics. This distinction
is recorded in the output. Result accumulation is streamed over test
batches to limit memory consumption.

```bash
python summarize_results.py
python summarize_results.py --output_dir outputs --csv outputs/results.csv
```

`--skip_completed` skips a run only after its `metrics.json` exists. The
runner refuses to overwrite an existing checkpoint. To retry an
interrupted run, preserve the previous artifacts and choose a new
`--output_dir`.

For test-only execution, repeat the original model and data arguments,
then add `--mode test --checkpoint <run-directory>/checkpoint.pth`.
The runner validates model configuration, column order, and training
normalization statistics before loading a checkpoint.

## Use the models directly

```python
import torch
from models import ModelConfig, ToMi, ToMiV

config = ModelConfig(
    window_size=24, seq_len=96, pred_len=192, enc_in=7,
    d_model=128, n_heads=8, num_hidden_layers=1,
)
model = ToMi(config)  # Or ToMiV(config)
x = torch.randn(2, 96, 7)
y = model(x)         # [2, 192, 7]
```

## Checks and code structure

```bash
python -m unittest discover -s tests -v
```

Tests cover the requested window and horizon ranges, padding, gradients,
gate behavior, training, checkpoint evaluation, result summaries, and
completed-run skipping. The end-to-end test creates temporary synthetic
data and deletes it afterwards.

See [docs/CODE_STRUCTURE.md](docs/CODE_STRUCTURE.md) for the code map and
tensor shapes. RevIN is an existing normalization method; its use should
be acknowledged alongside this work and the datasets.
