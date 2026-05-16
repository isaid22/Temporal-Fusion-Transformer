import os
import sys
import json
import argparse
import inspect
import pandas as pd
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
import yaml

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.search.hyperopt import HyperOptSearch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from atm_cash_forecast.config import config as project_config
from atm_cash_forecast.src.model import build_dataset
import torch

# Ensure GPU is available and set precision
torch.set_float32_matmul_precision('high')
if not torch.cuda.is_available():
    print("Error: CUDA is not available. Ray Tune for this model requires a GPU.")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Run hyperparameter tuning for TFT.")
    parser.add_argument(
        "--search-space-config",
        default=os.path.join(os.path.dirname(__file__), "tune_search_space.yaml"),
        help="Path to YAML or JSON file defining Ray Tune search space.",
    )
    return parser.parse_args()


def _build_tune_distribution(param_name, spec):
    kind = spec.get("type")
    if kind == "choice":
        return tune.choice(spec["values"])
    if kind == "uniform":
        return tune.uniform(spec["lower"], spec["upper"])
    if kind == "loguniform":
        return tune.loguniform(spec["lower"], spec["upper"])
    if kind == "randint":
        return tune.randint(spec["lower"], spec["upper"])
    raise ValueError(f"Unsupported search-space type '{kind}' for parameter '{param_name}'.")


def load_tuning_spec(spec_path):
    if not os.path.exists(spec_path):
        raise FileNotFoundError(f"Search-space config not found: {spec_path}")

    ext = os.path.splitext(spec_path)[1].lower()
    with open(spec_path, "r", encoding="utf-8") as f:
        if ext == ".json":
            spec = json.load(f)
        else:
            spec = yaml.safe_load(f)

    if not isinstance(spec, dict) or "search_space" not in spec:
        raise ValueError("Search-space config must contain a top-level 'search_space' object.")

    raw_space = spec["search_space"]
    if not isinstance(raw_space, dict) or not raw_space:
        raise ValueError("'search_space' must be a non-empty object.")

    search_space = {
        name: _build_tune_distribution(name, param_spec)
        for name, param_spec in raw_space.items()
    }

    num_samples = int(spec.get("num_samples", 20))
    resources_per_trial = spec.get("resources_per_trial", {"cpu": 4, "gpu": 1})
    if not isinstance(resources_per_trial, dict):
        raise ValueError("'resources_per_trial' must be an object with cpu/gpu fields.")

    return {
        "search_space": search_space,
        "num_samples": num_samples,
        "resources_per_trial": resources_per_trial,
        "parameter_names": list(raw_space.keys()),
    }


def split_trial_config(tune_config):
    """
    Split a Ray trial config into dataloader/model/trainer overrides.
    This allows adding new searchable parameters without code edits,
    as long as names match Lightning Trainer or TFT from_dataset kwargs.
    """
    dataloader_keys = {"batch_size", "val_batch_size_factor", "num_workers", "pin_memory"}
    alias_map = {
        "hidden_dim": "hidden_size",
        "embedding_dim": "hidden_continuous_size",
        "num_heads": "attention_head_size",
    }

    trainer_allowed = set(inspect.signature(pl.Trainer.__init__).parameters.keys())
    model_allowed = set(inspect.signature(TemporalFusionTransformer.from_dataset).parameters.keys())

    dataloader_cfg = {
        "batch_size": int(tune_config.get("batch_size", project_config.batch_size)),
        "val_batch_size_factor": int(tune_config.get("val_batch_size_factor", 2)),
        "num_workers": int(tune_config.get("num_workers", 2)),
        "pin_memory": bool(tune_config.get("pin_memory", torch.cuda.is_available())),
    }

    model_overrides = {}
    trainer_overrides = {}
    unknown_keys = []

    for key, value in tune_config.items():
        if key in dataloader_keys:
            continue
        mapped = alias_map.get(key, key)
        if mapped in trainer_allowed:
            trainer_overrides[mapped] = value
        elif mapped in model_allowed:
            model_overrides[mapped] = value
        else:
            unknown_keys.append(key)

    if unknown_keys:
        allowed_names = sorted(dataloader_keys | set(alias_map.keys()) | trainer_allowed | model_allowed)
        raise ValueError(
            "Unsupported search-space parameter(s): "
            f"{unknown_keys}.\n"
            "Use dataloader keys, Trainer kwargs, or "
            "TemporalFusionTransformer.from_dataset kwargs.\n"
            f"Examples of supported names: {allowed_names[:25]} ..."
        )

    return dataloader_cfg, model_overrides, trainer_overrides

def training_function(tune_config, train_dataset=None, val_dataset=None):
    """
    This function is called by Ray Tune for each trial.
    """
    # 1. Split trial config into known targets
    dataloader_cfg, model_overrides, trainer_overrides = split_trial_config(tune_config)

    # 2. Create DataLoaders
    train_dataloader = train_dataset.to_dataloader(
        train=True,
        batch_size=dataloader_cfg["batch_size"],
        num_workers=dataloader_cfg["num_workers"],
        pin_memory=dataloader_cfg["pin_memory"],
    )
    val_dataloader = val_dataset.to_dataloader(
        train=False,
        batch_size=dataloader_cfg["batch_size"] * dataloader_cfg["val_batch_size_factor"],
        num_workers=dataloader_cfg["num_workers"],
        pin_memory=dataloader_cfg["pin_memory"],
    )

    # 3. Initialize TFT Model with trial's hyperparameters
    model_kwargs = {
        "learning_rate": project_config.learning_rate,
        "hidden_size": project_config.hidden_dim,
        "attention_head_size": project_config.num_heads,
        "dropout": project_config.dropout,
        "hidden_continuous_size": project_config.embedding_dim,
        "output_size": 7,
        "loss": QuantileLoss(),
        "log_interval": 5,
        "reduce_on_plateau_patience": 4,
    }
    model_kwargs.update(model_overrides)

    tft = TemporalFusionTransformer.from_dataset(
        train_dataset,
        **model_kwargs,
    )

    # Workaround for mixed-precision bug
    for module in tft.modules():
        if hasattr(module, "mask_bias"):
            module.mask_bias = -1e4

    # 4. Configure Callbacks for this trial
        # Report metrics back to Ray Tune
    tune_report_callback = TuneReportCallback({"val_loss": "val_loss"}, on="validation_end")
    
    # Early stopping for this trial
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=5, mode="min")

    # 5. Create a PyTorch Lightning Trainer
    trainer_kwargs = {
        "max_epochs": project_config.epochs,
        "accelerator": "gpu",
        "devices": 1,
        # Mixed precision can be unstable for some trial configs in this stack.
        "precision": "32-true",
        "gradient_clip_val": 0.1,
        "gradient_clip_algorithm": "norm",
        "callbacks": [early_stop_callback, tune_report_callback],
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "num_sanity_val_steps": 0,
    }
    trainer_kwargs.update(trainer_overrides)

    trainer = pl.Trainer(
        **trainer_kwargs,
    )

    # 6. Fit the model
    trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


def run_tuning(search_space_config_path):
    print("="*60)
    print("STARTING HYPERPARAMETER SEARCH WITH RAY TUNE")
    print("="*60)

    # 1. Load and prepare data ONCE to be shared across all trials
    print("Loading and preparing data...")
    df = pd.read_csv(f"{project_config.data_path}/train_features.csv")
    df['atm_id'] = df['atm_id'].astype(str)
    df['city'] = df['city'].astype(str)
    df['day_of_week'] = df['day_of_week'].astype(str)
    df['month'] = df['month'].astype(str)

    max_time_idx = df["time_idx"].max()
    training_cutoff = max_time_idx - project_config.forecast_days

    train_dataset = build_dataset(
        df[df['time_idx'] <= training_cutoff],
        max_encoder_length=project_config.history_days,
        max_prediction_length=project_config.forecast_days
    )
    val_dataset = TimeSeriesDataSet.from_dataset(train_dataset, df, predict=True, stop_randomization=True)

    # 2. Load hyperparameter search space from external config
    print(f"Loading search space from: {search_space_config_path}")
    tuning_spec = load_tuning_spec(search_space_config_path)
    search_space = tuning_spec["search_space"]

    # 3. Define the scheduler and search algorithm
    # ASHA prunes unpromising trials early
    scheduler = ASHAScheduler(
        metric="val_loss",
        mode="min",
        max_t=project_config.epochs,
        grace_period=5, # Min epochs before a trial can be stopped
        reduction_factor=2
    )
    
    # HyperOpt uses Bayesian Optimization to intelligently search the space
    search_alg = HyperOptSearch(metric="val_loss", mode="min")

    # 4. Define a reporter for clean command-line output
    reporter = CLIReporter(
        metric_columns=["val_loss", "training_iteration"],
        parameter_columns=tuning_spec["parameter_names"]
    )

    # 5. Run the tuning job
    results_root = os.path.join(os.getcwd(), "ray_results")
    os.makedirs(results_root, exist_ok=True)

    run_args = dict(
        run_or_experiment=tune.with_parameters(
            training_function,
            train_dataset=train_dataset,
            val_dataset=val_dataset
        ),
        resources_per_trial=tuning_spec["resources_per_trial"],
        config=search_space,
        num_samples=tuning_spec["num_samples"],
        scheduler=scheduler,
        search_alg=search_alg,
        progress_reporter=reporter,
        name="tft_atm_tune",
    )

    # Newer Ray versions prefer storage_path; older ones use local_dir.
    try:
        analysis = tune.run(storage_path=results_root, **run_args)
    except TypeError:
        analysis = tune.run(local_dir=results_root, **run_args)

    print("\n" + "="*60)
    print("HYPERPARAMETER SEARCH COMPLETE")
    print("="*60)
    best_trial = analysis.get_best_trial("val_loss", "min", "last")
    print(f"\n✅ Best trial found:")
    print(f"  - Validation Loss: {best_trial.last_result['val_loss']:.4f}")
    print(f"  - Config: {best_trial.config}")
    best_trial_dir = getattr(best_trial, "path", None) or getattr(best_trial, "logdir", None)
    all_trials_dir = os.path.join(results_root, "tft_atm_tune")
    if best_trial_dir:
        print(f"\nBest trial logs: {best_trial_dir}")
    print(f"All trial logs root: {all_trials_dir}")


if __name__ == "__main__":
    args = parse_args()
    run_tuning(args.search_space_config)
