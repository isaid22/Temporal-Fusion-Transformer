import os
import json
import numpy as np
import pandas as pd

def evaluate_forecasts(predictions, quantiles=[0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]):
    """
    Evaluates the model's accuracy comparing the median point forecast to actuals.
    Remember: Units are in 'number of bills', not dollars.
    """
    try:
        # Extract median prediction
        # predictions.output holds a tuple of outputs when return_x is true in raw mode.
        # The main network output is the first element of that tuple, shaped (batches, time, quantiles)
        main_output = predictions.output[0]
        
        median_idx = quantiles.index(0.5)
        # We must move the tensor from the GPU (cuda) back to the CPU before converting to NumPy
        median_pred = main_output[:, :, median_idx].cpu().numpy()
        
        # Ground truth actuals
        # predictions.y returns a tuple where the first element is the actual target tensor
        actuals = predictions.y[0].cpu().numpy()
    except AttributeError:
        print("Evaluation skipped: return_y=True must be passed to tft.predict().")
        return

    # Calculate errors (flattening the tensors so we evaluate across all ATMs and all days)
    errors = median_pred - actuals
    absolute_errors = np.abs(errors)
    squared_errors = errors ** 2

    # Point Metrics
    mae = np.mean(absolute_errors)
    rmse = np.sqrt(np.mean(squared_errors))
    
    # sMAPE (Symmetric Mean Absolute Percentage Error) - avoids div by zero on low counts
    # sMAPE = 200 * |F - A| / (|F| + |A|)
    denominator = np.abs(median_pred) + np.abs(actuals)
    
    # To prevent division by zero completely, add a small epsilon
    epsilon = 1e-8
    smape = np.mean(200 * absolute_errors / (denominator + epsilon))

    print("\n" + "="*50)
    print(" 📊 HOLDOUT EVALUATION METRICS (50th Percentile)")
    print("="*50)
    print(f"Unit: Number of Bills (not dollar amount)\n")
    print(f"MAE  (Mean Absolute Error):        {mae:.2f} bills per day")
    print(f"RMSE (Root Mean Squared Error):    {rmse:.2f} bills per day")
    print(f"sMAPE (Symmetric MAPE):            {smape:.2f} %")
    print("="*50 + "\n")
    
    # Save the metrics to disk for tracking and downstream reporting
    os.makedirs("results", exist_ok=True)
    metrics_dict = {
        "unit": "number_of_bills",
        "MAE": float(mae),
        "RMSE": float(rmse),
        "sMAPE_percent": float(smape)
    }
    
    metrics_path = "results/holdout_evaluation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=4)
        
    print(f"Metrics successfully saved to: {metrics_path}")
    
    # -------------------------------------------------------------
    # Export Raw Predictions for Manual Inspection & Post-Processing
    # -------------------------------------------------------------
    if hasattr(predictions, 'index'):
        # median_pred is shape (batch_size, forecast_days)
        batch_size, forecast_days = median_pred.shape
        
        # Flatten everything to 1D columns
        atm_ids = np.repeat(predictions.index['atm_id'].values, forecast_days)
        start_time_idxs = np.repeat(predictions.index['time_idx'].values, forecast_days)
        step_offsets = np.tile(np.arange(forecast_days), batch_size)
        
        explicit_time_idxs = start_time_idxs + step_offsets
        
        # Build the base dictionary with identifiers, actuals, and error
        raw_dict = {
            'atm_id': atm_ids,
            'time_idx': explicit_time_idxs,
            'forecast_day_offset': step_offsets + 1,  # e.g., Day 1 to 35
            'actual_bills': actuals.flatten(),
            'absolute_error': absolute_errors.flatten()
        }
        
        # Dynamically loop through all 7 quantiles to capture confidence intervals
        for i, q in enumerate(quantiles):
            # Extract and flatten each probability band directly from the GPU tensor
            quant_pred = main_output[:, :, i].cpu().numpy().flatten()
            raw_dict[f'predicted_q{q:.2f}'] = quant_pred
            
        df_raw = pd.DataFrame(raw_dict)
        
        csv_path = "results/raw_predictions.csv"
        df_raw.to_csv(csv_path, index=False)
        print(f"Raw observations and predictions exported to: {csv_path}")
