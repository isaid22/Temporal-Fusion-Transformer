# Hyperparameter Tuning Guide (Ray Tune)

This document explains how to run hyperparameter tuning for the TFT model and where to find outputs.

## Prerequisites

1. Activate your Python environment.
2. Run from the project root directory.
3. Ensure training features exist at `atm_cash_forecast/data/train_features.csv`.

## Search Space File

Tune parameters are defined in:

- `atm_cash_forecast/tune_search_space.yaml`

You can add, remove, or adjust parameter ranges there.

## Run Tuning

From project root:

```bash
source /home/xps15/ai_venv/bin/activate
cd /home/xps15/projects/TFT/github/Temporal-Fusion-Transformer
python atm_cash_forecast/tune.py
```

### Run With a Custom Search Space Config

```bash
python atm_cash_forecast/tune.py --search-space-config atm_cash_forecast/tune_search_space.yaml
```

The config can be YAML or JSON.

## Where Results Are Saved

The current tuning script writes results under your launch directory:

- `<current-working-directory>/ray_results/tft_atm_tune`

If you launch from project root, outputs go to:

- `ray_results/tft_atm_tune`

Each trial has its own subfolder, typically containing:

- `params.json`: trial hyperparameters
- `result.json`: JSON-lines metrics per iteration
- `progress.csv`: tabular trial metrics over iterations

## Reviewing Best Trial

At the end of tuning, the script prints:

- best validation loss
- best hyperparameter config
- best trial log directory
- all-trials root directory

## Quick Inspection Commands

List trial folders:

```bash
find ray_results/tft_atm_tune -maxdepth 1 -type d | sort
```

Show one trial's hyperparameters:

```bash
python -m json.tool ray_results/tft_atm_tune/<trial_folder>/params.json
```

Show the latest metric record for a trial:

```bash
tail -n 1 ray_results/tft_atm_tune/<trial_folder>/result.json | python -m json.tool
```

## Notes

- Older runs from earlier script versions may be under `~/ray_results/tft_atm_tune`.
- Current script behavior is to use the directory where you launch `tune.py`.
