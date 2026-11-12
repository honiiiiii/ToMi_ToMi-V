# Code structure

```text
run.py                         Command-line configuration and run orchestration
summarize_results.py           Paired MSE/MAE collection; optional CSV export
requirements.txt               Tested dependency versions
models/
  config.py                    Validated model configuration
  tomi.py                      ToMi public model class
  tomi_v.py                    ToMi-V public model class
  base.py                      Shared embedding, encoding, and forecasting path
  temporal.py                  Multi-head token mixing and temporal blocks
  variable.py                  Gated low-rank cross-variable mixing
layers/
  revin.py                     Reversible instance normalization
data_provider/
  series.py                    File reading, chronological splits, and windows
engine/
  loss.py                      Joint pointwise/spectral objective
  trainer.py                   Adam training, validation, checkpoints, evaluation
scripts/
  run_forecasting.sh           Four-horizon shell runner with failure continuation
tests/
  test_models.py               Model shape, gradient, padding, and gate checks
  test_smoke.py                Temporary-data training and evaluation checks
docs/
  CONFIGURATION.md             Argument definitions and validity constraints
  CODE_STRUCTURE.md            This file
```

## Tensor layout

Let B be batch size, C the number of variables, L input length, H forecast
length, E scalar-embedding width, D hidden width, and w region length.
The number of temporal regions is P = ceil(L / w).

| Stage | Shape |
| --- | --- |
| Input and instance normalization | `[B, L, C]` |
| Shared scalar embedding | `[B, C, L, E]` |
| Hidden projection in each temporal block | `[B, C, L, D]` |
| Right padding and temporal rearrangement | `[B*C, P, w, D]` |
| Multi-head mixing plus local depthwise convolution | `[B*C, P, w, D]` |
| Temporal restoration, padding removal, output projection | `[B, C, L, E]` |
| Block residual update | `[B, C, L, E]` |
| ToMi-V cross-variable residual mixing | `[B, C, L, E]` |
| Flatten time and embedding; shared feature projection | `[B, C, D]` |
| Two-layer forecasting MLP | `[B, C, H]` |
| Axis restoration and inverse instance normalization | `[B, H, C]` |

ToMi applies the temporal encoder independently to each variable with
shared weights. ToMi-V inserts one low-rank variable mixer after the
final temporal block and before the forecasting head.

## Temporal mixing

`MultiHeadTokenMixer` forms nonnegative query and key features using ReLU
plus a numerical epsilon. Each temporal region produces a compact
key-value representation. `RegionMixer` learns region-to-region
interactions through a shared matrix, initialized from column-normalized
cosine distance weights. The module applies the same learned matrix in
the output and normalization branches. The local branch is a depthwise
convolution on the temporal region grid. Outputs are combined and
projected before the block residual update.

The implementation preserves the operations and padding convention of
the model. Padding is removed after restoring the temporal layout; it
does not create additional forecast targets. A token-to-token attention
matrix over all temporal positions is not materialized.

## Cross-variable mixing

`LowRankVariableMixer` projects variables from C to r latent components,
applies GELU, and projects back to C. Both projections are shared across
time and embedding coordinates. The effective rank is `min(requested
rank, C)`. A scalar `tanh` gate controls the dropout-regularized residual
contribution. A zero-initialized gate initially preserves the temporal
representation; its value is learned during optimization.

## Source boundaries

Model classes depend only on PyTorch, einops, and the local RevIN layer.
Data handling and training are separate modules. No baseline registry,
external repository import, network download, machine-specific path,
or experiment-history lookup is needed.
