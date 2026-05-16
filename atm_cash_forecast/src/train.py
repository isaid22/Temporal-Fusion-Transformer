import os
import sys
import pandas as pd
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss
from torch.utils.data import DataLoader
import torch
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import config
from src.model import build_dataset
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train TFT Model for ATM Cash Forecast")
    
    # Add arguments for hyperparameter tuning
    parser.add_argument("--hidden_dim", type=int, default=config.hidden_dim, help="Size of hidden dimensions")
    parser.add_argument("--num_heads", type=int, default=config.num_heads, help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=config.dropout, help="Dropout rate")
    parser.add_argument("--learning_rate", type=float, default=config.learning_rate, help="Learning rate")
    parser.add_argument("--history_days", type=int, default=config.history_days, help="Lookback window")
    parser.add_argument("--batch_size", type=int, default=config.batch_size, help="Training batch size")
    parser.add_argument("--val_batch_size_factor", type=int, default=2, help="Validation batch size multiplier")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader worker processes")
    parser.add_argument("--prefetch_factor", type=int, default=4, help="Batches prefetched per worker")
    parser.add_argument("--persistent_workers", action="store_true", help="Keep DataLoader workers alive between epochs")
    parser.add_argument("--no_pin_memory", action="store_true", help="Disable pinned host memory for DataLoader")
    parser.add_argument(
        "--precision",
        type=str,
        default="32-true",
        choices=["32-true", "16-mixed", "bf16-mixed"],
        help="Training precision. Use 32-true for stability, 16/bf16-mixed for speed.",
    )
    
    args = parser.parse_args()
    
    # Override config with parsed arguments
    config.hidden_dim = args.hidden_dim
    config.num_heads = args.num_heads
    config.dropout = args.dropout
    config.learning_rate = args.learning_rate
    config.history_days = args.history_days
    config.batch_size = args.batch_size
    
    return args

def train_tft():
    # Parse hyperparameter overrides from command line
    args = parse_args()
    
    print("="*60)
    print("TRAINING TEMPORAL FUSION TRANSFORMER (GPU)")
    print("="*60)
    
    # 1. Load Data
    train_path = f"{config.data_path}/train_features.csv"
    if not os.path.exists(train_path):
        print(f"Error: {train_path} missing. Run feature_engineering.py first.")
        return
        
    print(f"Loading engineered training data from {train_path}...")
    df = pd.read_csv(train_path)
    
    # PyTorch Forecasting requires purely string variables for its categorical embeddings
    df['atm_id'] = df['atm_id'].astype(str)
    df['city'] = df['city'].astype(str)
    df['day_of_week'] = df['day_of_week'].astype(str)
    df['month'] = df['month'].astype(str)
    
    # Validation split (let's use the last forecast_days of the training set for validation)
    max_time_idx = df["time_idx"].max()
    training_cutoff = max_time_idx - config.forecast_days
    
    # 2. Create Datasets
    print("Building PyTorch Forecasting Dataset Definitions...")
    train_dataset = build_dataset(
        df[df['time_idx'] <= training_cutoff], 
        max_encoder_length=config.history_days, 
        max_prediction_length=config.forecast_days
    )
    
    val_dataset = TimeSeriesDataSet.from_dataset(
        train_dataset, 
        df, 
        predict=True, 
        stop_randomization=True
    )
    
    # 3. Create DataLoaders
    batch_size = config.batch_size
    pin_memory = (not args.no_pin_memory) and torch.cuda.is_available()
    loader_kwargs = {
        "num_workers": max(0, args.num_workers),
        "pin_memory": pin_memory,
    }
    if loader_kwargs["num_workers"] > 0:
        loader_kwargs["persistent_workers"] = args.persistent_workers
        loader_kwargs["prefetch_factor"] = max(2, args.prefetch_factor)

    train_dataloader = train_dataset.to_dataloader(
        train=True,
        batch_size=batch_size,
        **loader_kwargs,
    )
    val_dataloader = val_dataset.to_dataloader(
        train=False,
        batch_size=batch_size * max(1, args.val_batch_size_factor),
        **loader_kwargs,
    )
    
    # 4. Initialize TFT Model
    print("\nInitializing Temporal Fusion Transformer Architecture...")
    tft = TemporalFusionTransformer.from_dataset(
        train_dataset,
        learning_rate=config.learning_rate,
        hidden_size=config.hidden_dim,
        attention_head_size=config.num_heads,
        dropout=config.dropout,
        hidden_continuous_size=config.embedding_dim,
        output_size=7,  # 7 quantiles for default QuantileLoss
        loss=QuantileLoss(), 
        log_interval=10, 
        reduce_on_plateau_patience=4,
    )
    
    # 5. Configure Training Options
    os.makedirs(config.model_save_path, exist_ok=True)

    # Workaround for mixed-precision bug in some pytorch-forecasting versions
    # The InterpretableMultiHeadAttention submodule has a hard-coded -1e9 mask bias
    # which overflows with 16-bit precision. We need to manually overwrite it.
    for module in tft.modules():
        if hasattr(module, "mask_bias"):
            module.mask_bias = -1e4
    
    # 5. Configure Callbacks and Logging
    # Stop training early if validation loss stops improving
    early_stop_callback = EarlyStopping(
        monitor="val_loss", 
        min_delta=1e-4, 
        patience=config.early_stopping_patience, 
        verbose=True, 
        mode="min"
    )
    
    # Checkpoint callback: save the best model weights dynamically
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.model_save_path,
        filename="tft_epoch-{epoch:02d}_val-loss-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        auto_insert_metric_name=False,  # This prevents the annoying '=' signs
    )
    
    # Set up TensorBoard
    logger = TensorBoardLogger("lightning_logs", name="tft_atm_cash_forecast")
    
    # Log hyperparameters to TensorBoard
    from dataclasses import asdict
    logger.log_hyperparams(asdict(config))
    
    def build_trainer(precision_mode: str) -> pl.Trainer:
        return pl.Trainer(
            max_epochs=config.epochs,
            accelerator="gpu",   # This uses your RTX 4060
            devices=1,
            precision=precision_mode,
            gradient_clip_val=0.1,
            gradient_clip_algorithm="norm",
            logger=logger,       # Automatically log all steps/epochs metrics here
            callbacks=[early_stop_callback, checkpoint_callback, RichProgressBar()],
            enable_model_summary=True,
            log_every_n_steps=10,
            num_sanity_val_steps=0  # Skip pre-epoch validation check to prevent hanging
        )

    trainer = build_trainer(args.precision)

    # 6. Fit Model
    print(f"\nIgniting GPU Training Loop for {config.epochs} max epochs...")
    try:
        trainer.fit(
            tft,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )
    except RuntimeError as exc:
        err_msg = str(exc)
        should_fallback = (
            args.precision == "16-mixed"
            and "does not require grad" in err_msg
            and "grad_fn" in err_msg
        )
        if not should_fallback:
            raise

        print("\nAMP training failed with a known autograd issue.")
        print("Retrying automatically with precision=32-true for stability...")

        trainer = build_trainer("32-true")
        trainer.fit(
            tft,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )
    
    print("\nTraining Complete! Saving best model weights...")
    best_model_path = trainer.checkpoint_callback.best_model_path
    print(f"Best model saved at: {best_model_path}")

if __name__ == "__main__":
    import torch
    print(f"CUDA Available: {torch.cuda.is_available()}")
    train_tft()
