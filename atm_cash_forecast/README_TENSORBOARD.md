# TensorBoard Guide

This guide explains how to launch TensorBoard for training and tuning runs in this project.

## Prerequisites

1. Activate your Python environment.
2. Run commands from project root.

```bash
source /home/xps15/ai_venv/bin/activate
cd /home/xps15/projects/TFT/github/Temporal-Fusion-Transformer
```

## Training Logs (PyTorch Lightning)

Training logs are written under:

- `atm_cash_forecast/lightning_logs`

Launch TensorBoard:

```bash
tensorboard --logdir atm_cash_forecast/lightning_logs --port 6006
```

Open in browser:

- http://localhost:6006

## Tuning Logs (Ray Tune)

Hyperparameter tuning logs are written under the run launch directory:

- `ray_results/tft_atm_tune`

Launch TensorBoard for tuning runs:

```bash
tensorboard --logdir ray_results/tft_atm_tune --port 6007
```

Open in browser:

- http://localhost:6007

## Compare Training and Tuning Together

You can point TensorBoard to both trees:

```bash
tensorboard \
  --logdir_spec train:atm_cash_forecast/lightning_logs,tune:ray_results/tft_atm_tune \
  --port 6008
```

## Helpful Tips

1. If port is in use, choose another port (for example 6009).
2. If a previous TensorBoard instance is stale, stop it and relaunch.
3. For remote machine access, bind to all interfaces only if needed:

```bash
tensorboard --logdir atm_cash_forecast/lightning_logs --host 0.0.0.0 --port 6006
```

## What to Inspect

1. Scalars: `train_loss_epoch` and `val_loss` trends.
2. HPARAMS: compare trial hyperparameters and outcomes.
3. Step axis: interpreted as global steps, not epochs.
