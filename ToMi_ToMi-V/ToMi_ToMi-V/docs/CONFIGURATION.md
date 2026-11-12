# Configuration

All model and task lengths are supplied through `ModelConfig` or CLI
arguments. `window_size` has no implicit default.

| Argument | Meaning | Default / constraint |
| --- | --- | --- |
| `--model` | Model implementation | Required: `tomi` or `tomi-v` |
| `--data` | Dataset split protocol | Required; see README |
| `--root_path` | Prefix for data path | `.` |
| `--data_path` | Dataset file | Required |
| `--features` | All-variable, target-only output, or single-series task | `M`; also `MS`, `S` |
| `--target` | Named target column | Required for `MS` / `S` |
| `--seq_len` | Number of observed time steps L | 96; positive |
| `--pred_len` | Forecast horizon H | 96; positive |
| `--window_size` | Temporal region length w | Required; 1 <= w <= L |
| `--enc_in` | Number of input variables C | Inferred; supplied values are checked |
| `--embed_size` | Scalar embedding width E | 16 |
| `--d_model` | Hidden temporal width D | 128; divisible by `n_heads` |
| `--n_heads` | Number of feature heads | 8 |
| `--num_hidden_layers` | Number of temporal blocks | 1 |
| `--d_ff` | Forecasting MLP hidden width | 1024 |
| `--dropout` | Temporal branch dropout | 0.1; in [0, 1) |
| `--local_kernel_size` | Local depthwise convolution size | 5; positive odd integer |
| `--channel_mixer_rank` | ToMi-V latent variable rank | 4; capped at C |
| `--channel_mixer_dropout` | ToMi-V branch dropout | Inherits `dropout` |
| `--channel_mixer_gate_init` | ToMi-V gate initialization | 0 |
| `--alpha` | Spectral contribution to joint loss | 0.3; in [0, 1] |
| `--learning_rate` | Initial Adam learning rate | 0.001 |
| `--batch_size` | Training and validation batch size | 32 |
| `--eval_batch_size` | Test batch size | 1 |
| `--train_epochs` | Maximum training epochs | 60 |
| `--patience` | Validation epochs without improvement | 3 |
| `--lradj` | Learning-rate schedule | `type1`; also `constant` |
| `--train_ratio` | Non-ETT training fraction | 0.7 |
| `--test_ratio` | Non-ETT test fraction | 0.2 |
| `--seed` | Random seed | 2021 |
| `--num_workers` | DataLoader worker count | 0 |
| `--num_threads` | PyTorch CPU thread count | 2 |
| `--device` | Compute device | `auto`; also `cpu`, `cuda:0`, ... |
| `--output_dir` | Result directory | `outputs` |
| `--mode` | Train-and-test or checkpoint-only test | `train`; also `test` |
| `--checkpoint` | Checkpoint for test mode | Required in test mode |
| `--inverse` | Report metrics in original dataset units | Off |
| `--skip_completed` | Skip identical completed configurations | Off; enabled by shell runner |

The `type1` schedule updates the learning rate after each epoch. For
epoch e, the rate is the initial rate when e=1 and e=2, then the initial
rate times `0.5 ** (e-2)` for e>=3. The loss always includes both time
terms unless their combined coefficient is zero. To disable the spectral
term, use `--alpha 0`.

Changing L or w changes temporal region layout and can change the shape
of the learned region-mixing matrix. A checkpoint must therefore be
loaded using its saved model configuration. Changing H changes the
forecasting head output dimension. No architectural parameter is
silently adjusted to accommodate incompatible dimensions.
