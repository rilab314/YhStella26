# STELLA — Lane Marking Detection from Satellite Imagery

STELLA detects lane markings in satellite image tiles and outputs each marking as a separate
polyline with a class label. It is the code of the paper
*"STELLA: A Connectivity Representation for Lane Marking Detection from Satellite Imagery"*.

At each cell of a grid over the image, the model predicts one **self slot** and two
**connection slots**. The self slot describes the cell: whether a line passes through it, the class
of that line, and its exact position inside the cell. Each connection slot gives one direction in
which the line continues. The two connection slots have no order, and they carry a direction only,
not a distance. A decoder with no learned parameters links slots that face each other into chains,
and each chain becomes one polyline. The ground truth is encoded in the same form, so one line
corresponds to one chain of cells.

## Results

Test set of SEED-MAP v1.2 (2,457 tiles). Instance F1 counts a predicted line as correct only when
it matches one ground-truth line of the same class.

| Method | Instance F1 | Precision | Recall |
| --- | --- | --- | --- |
| Segmentation + skeletonization + Douglas–Peucker | 0.3005 | 0.2206 | 0.4713 |
| Sat2Graph output representation (angular sectors + displacement) | 0.3551 | 0.3634 | 0.3472 |
| **STELLA** | **0.4483** | **0.4733** | **0.4258** |

The Sat2Graph row uses the same backbone, neck, grid, and training budget as STELLA.
Only the output representation is different.

## Installation

The environment is managed with [uv](https://docs.astral.sh/uv/).
Python 3.11, PyTorch 2.11 (CUDA 12.8), and PyTorch Lightning are pinned in `uv.lock`.

```bash
git clone https://github.com/rilab314/YhStella26.git
cd YhStella26
uv sync --extra dev --frozen
.venv/bin/python -m pytest tests/ -q -m "not slow"   # quick check
```

PyTorch 2.7 or later is required for NVIDIA Blackwell GPUs (sm_120).

## Data

All data paths are set by one environment variable. Choose a folder and set it before running
any script:

```bash
export STELLA_DATA_ROOT=/path/to/stella_data
```

The scripts then use the following layout.

```
$STELLA_DATA_ROOT/
  SEED_MAP_v1.2_splits/{train,val,test}/{image,label}   dataset (built in step 2)
  gt_cache/                                             encoded ground truth (created automatically)
  pred_cache/                                           saved model predictions
  paper/score/                                          scores written by the evaluation script
  log/                                                  training runs
```

**1. Download SEED-MAP v1.2.** The dataset is published in the SEED-MAP repository:
<https://github.com/rilab314/SatelliteLaneDataset2024>.
It contains flat `image/` and `label/` folders and a `dataset.json` file with the split.

**2. Arrange it into split folders.** STELLA reads the dataset as
`{train,val,test}/{image,label}`. The script copies the files into this layout without changing them.

```bash
.venv/bin/python scripts/build_split_dataset.py \
    --src /path/to/SEED_MAP_v1.2 \
    --dst $STELLA_DATA_ROOT/SEED_MAP_v1.2_splits
```

The split has 8,979 training, 1,218 validation, and 2,457 test tiles. The tile lists are included in
the reproduction package below (`splits/*.txt`) so that you can check your copy.

## Reproduction package

**Download:** [stella_reproduce.zip (1.6 GB, OneDrive)](https://1drv.ms/u/c/f3539410ca672e92/IQD5nRkFnCIMTLed5QFFYKt4AWC9m7fMk7ibWPaknM4Dh2k?e=Y9lveG)

| Path in the archive | Contents |
| --- | --- |
| `checkpoints/stella_final/` | STELLA final model (`model.ckpt`, weights only) and its training configuration (`config.json`) |
| `checkpoints/sat2graph_gte_delta/` | Sat2Graph baseline model and its training configuration |
| `splits/` | Tile names of the training, validation, and test sets |
| `baselines/segmentation_predictions/` | Predictions of the segmentation baseline (color-coded PNG per tile) |
| `expected_scores.json` | Scores that a correct reproduction gives |

The dataset images are not in the archive. Download them as described in [Data](#data).

## Reproducing the results

Unzip the archive and set its location.

```bash
export PKG=/path/to/stella_reproduce
```

Each model is evaluated in two steps. The first step runs the network on the GPU and saves its
outputs. The second step decodes the outputs into polylines and scores them on the CPU.
The `config.json` files keep the data paths of our machine, so the first step overrides them.

**STELLA**

```bash
.venv/bin/python scripts/dump_predictions.py --config configs.base \
    --run $PKG/checkpoints/stella_final --ckpt $PKG/checkpoints/stella_final/model.ckpt \
    --split test --count 2457 --out $STELLA_DATA_ROOT/pred_cache/stella_test \
    --override data.root=$STELLA_DATA_ROOT/SEED_MAP_v1.2_splits data.cache_dir=$STELLA_DATA_ROOT/gt_cache
.venv/bin/python scripts/eval_methods.py --method stella --split test \
    --cache $STELLA_DATA_ROOT/pred_cache/stella_test
```

**Sat2Graph baseline**

```bash
.venv/bin/python scripts/dump_predictions.py --config configs.exp_gte_delta \
    --run $PKG/checkpoints/sat2graph_gte_delta --ckpt $PKG/checkpoints/sat2graph_gte_delta/model.ckpt \
    --split test --count 2457 --out $STELLA_DATA_ROOT/pred_cache/sat2graph_test \
    --override data.root=$STELLA_DATA_ROOT/SEED_MAP_v1.2_splits data.cache_dir=$STELLA_DATA_ROOT/gt_cache
.venv/bin/python scripts/eval_methods.py --method gte_delta --split test \
    --cache $STELLA_DATA_ROOT/pred_cache/sat2graph_test
```

**Segmentation baseline** (no network to run; the predictions are in the archive)

```bash
.venv/bin/python scripts/eval_methods.py --method seg_dp --split test \
    --pred $PKG/baselines/segmentation_predictions/pred_test
```

Each evaluation writes `$STELLA_DATA_ROOT/paper/score/<method>_test/summary.json`.
Compare its `f1`, `precision`, and `recall` with `expected_scores.json`.
On one RTX PRO 6000 GPU, the first step takes about 4 minutes per model.
The second step takes about 1 to 1.5 hours per method on one CPU core.

## Training

```bash
.venv/bin/python -m stella.train.train --config configs.base --tag myrun
```

- `--config` selects a configuration module in `configs/` (`configs.base` is the final model,
  `configs.exp_gte_delta` is the Sat2Graph baseline).
- `--override key=value ...` changes configuration fields, for example `train.epochs=40`.
- Each run creates `$STELLA_DATA_ROOT/log/<date>_<config>_<tag>/` with its configuration,
  a copy of the source code, checkpoints, and `metrics.csv`.

The final model was trained for 40 epochs on two GPUs (`--override train.devices=2`).
On some multi-GPU machines without NVLink, add `NCCL_P2P_DISABLE=1` before the command
if training stops at the first step.

## Repository layout

```
configs/    experiment configurations (schema.py defines every field)
stella/     data encoding, model, losses, decoder, and evaluation
scripts/    data preparation, prediction, evaluation, tables, and figures
tests/      unit tests
docs/       design notes (Korean)
```
