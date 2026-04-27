import os
import sys
import pandas as pd
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss
from torch.utils.data import DataLoader

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
    
    args = parser.parse_args()
    
    # Override config with parsed arguments
    config.hidden_dim = args.hidden_dim
    config.num_heads = args.num_heads
    config.dropout = args.dropout
    config.learning_rate = args.learning_rate
    config.history_days = args.history_days
    
    return args

def train_tft():
    # Parse hyperparameter overrides from command line
    parse_args()
    
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
    train_dataloader = train_dataset.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = val_dataset.to_dataloader(train=False, batch_size=batch_size * 2, num_workers=0)
    
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
        filename="tft-{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        auto_insert_metric_name=True,
    )
    
    # Set up TensorBoard
    logger = TensorBoardLogger("lightning_logs", name="tft_atm_cash_forecast")
    
    # Log hyperparameters to TensorBoard
    from dataclasses import asdict
    logger.log_hyperparams(asdict(config))
    
    trainer = pl.Trainer(
        max_epochs=config.epochs,
        accelerator="gpu",   # This uses your RTX 4060
        devices=1,
        gradient_clip_val=0.1,
        logger=logger,       # Automatically log all steps/epochs metrics here
        callbacks=[early_stop_callback, checkpoint_callback],
        enable_model_summary=True,
        log_every_n_steps=10,
        num_sanity_val_steps=0  # Skip pre-epoch validation check to prevent hanging
    )

    # 6. Fit Model
    print(f"\nIgniting GPU Training Loop for {config.epochs} max epochs...")
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
