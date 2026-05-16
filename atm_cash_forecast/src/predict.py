import os
import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add parent directory to path to allow importing local modules
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pytorch_forecasting import TemporalFusionTransformer
from src.evaluate import evaluate_forecasts
from config import config

def run_inference():
    print("="*60)
    print("STARTING TFT INFERENCE ON HOLDOUT DATA")
    print("="*60)
    
    # 1. Find the best model checkpoint
    ckpt_files = glob.glob(f"{config.model_save_path}/*.ckpt")
    if not ckpt_files:
        print("Error: No trained model checkpoints found in data/models/")
        return
        
    # Get the latest checkpoint
    best_ckpt = max(ckpt_files, key=os.path.getmtime)
    print(f"Loading weights from: {best_ckpt}")
    
    # Load model in evaluation mode
    tft = TemporalFusionTransformer.load_from_checkpoint(best_ckpt)
    tft.eval()

    # 2. Load the data
    train_path = f"{config.data_path}/train_features.csv"
    holdout_path = f"{config.data_path}/holdout_features.csv"
    
    print("Loading historical context and hold-out data...")
    train_df = pd.read_csv(train_path)
    holdout_df = pd.read_csv(holdout_path)
    
    # Keep categorical columns as strings
    for df in [train_df, holdout_df]:
        df['atm_id'] = df['atm_id'].astype(str)
        df['city'] = df['city'].astype(str)
        df['day_of_week'] = df['day_of_week'].astype(str)
        df['month'] = df['month'].astype(str)

    # 3. Stitch Context + Target Horizon
    # We need the last 'history_days' from training as context
    max_train_idx = train_df['time_idx'].max()
    context_df = train_df[train_df['time_idx'] > max_train_idx - config.history_days].copy()
    
    # IMPORTANT: The holdout_df time_idx starts at 0 because it was generated separately.
    # We must offset it so it continues seamlessly from train_df's timeline.
    holdout_df['time_idx'] = holdout_df['time_idx'] + max_train_idx + 1
    
    # And we need the first 'forecast_days' from holdout as our target validation
    target_df = holdout_df[holdout_df['time_idx'] <= max_train_idx + config.forecast_days].copy()
    
    # Combine them into a continuous timeline for the model
    predict_df = pd.concat([context_df, target_df], ignore_index=True)

    # 4. Run Predictions
    print(f"Generating predictions for {config.forecast_days} days into Q1 2025...")
    # Using mode="raw" returns the exact dictionary format that plot_prediction() requires
    predictions = tft.predict(predict_df, mode="raw", return_x=True, return_y=True, return_index=True)
    
    # 5. Global Evaluation
    evaluate_forecasts(predictions)
    
    # 6. Visualize Results against Ground Truth
    os.makedirs("results", exist_ok=True)
    print("\nPlotting probabilistic forecasts...")
    
    # Let's plot the first 3 ATMs to see how we did
    for idx in range(3):
        # TFT's plot_prediction automatically draws the context, actuals (y), and 7 predicted quantiles
        fig, ax = plt.subplots(figsize=(12, 6))
        tft.plot_prediction(predictions.x, predictions.output, idx=idx, add_loss_to_title=True, ax=ax)
        
        # Custom legend
        legend_patches = [
            mpatches.Patch(color='grey', label='Historical Data'),
            mpatches.Patch(color='blue', label='Actuals'),
            mpatches.Patch(color='orange', label='Median Forecast'),
            mpatches.Patch(color='red', alpha=0.3, label='95% Confidence Interval'),
            mpatches.Patch(color='red', alpha=0.5, label='80% Confidence Interval')
        ]
        ax.legend(handles=legend_patches)
        
        atm_id = predict_df['atm_id'].unique()[idx]
        # Combine our custom title with the auto-generated loss metrics and add padding
        original_title = ax.get_title()
        ax.set_title(f"ATM {atm_id} - Cash Demand Forecast vs Actuals (Q1 2025)\n{original_title}", pad=20, fontsize=13)
        
        plt.tight_layout()
        
        save_path = f"results/forecast_atm_{atm_id}.png"
        plt.savefig(save_path)
        plt.close()
        print(f"Saved plot: {save_path}")

    print("\nInference complete! Check the 'results' folder for the plots.")

if __name__ == "__main__":
    run_inference()
