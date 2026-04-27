# Temporal-Fusion-Transformer
This is a repo for multiple examples of time series problems using Temporal Fusion Transformers.

## Project 1: ATM Cash Demand Forecasting

This project demonstrates an end-to-end framework for forecasting multi-horizon daily cash demand across a simulated network of 100 ATMs distributed across 4 major US cities. By leveraging the Temporal Fusion Transformer (TFT), the model identifies complex time-varying relationships (like payday cycles and holiday spikes) alongside static infrastructure covariates (like ATM location).

### 🏗️ Model Structure

This implementation uses the **Temporal Fusion Transformer (TFT)** architecture via the `pytorch-forecasting` library. The network uses multi-head attention to select relevant historical patterns, while independently handling:

*   **Static Covariates:** ATM ID, City (constant over time).
*   **Time-Varying Known Covariates:** Day of the week, Month, Year, Days until next holiday (future knowledge available).
*   **Time-Varying Unknown Covariates:** Past daily withdrawal amounts, past deposit amounts, rolling averages (only known up to the current day).

**Output:** Instead of a single point estimate, the model outputs **7 target quantiles** (probabilistic intervals: 2%, 10%, 25%, 50%, 75%, 90%, 98%) using **QuantileLoss**. This provides confidence bounds for cash forecasting, crucial for conservative physical logistics planning.

### ⚙️ Training and Validation Setup

*   **Backend:** PyTorch Lightning with automatic GPU acceleration (`accelerator="gpu"`).
*   **Early Stopping:** Monitors `val_loss` and halts training after `patience=10` epochs if the loss fails to improve, preventing overfitting.
*   **Sliding Windows:** Converts time-series into overlapping sliding windows of context. 
    *   *History Days (Encoder):* 90 days.
    *   *Forecast Days (Decoder):* 35 days horizon.
*   **Data Splits:** Synthetically generates historical data (2022-2024) for training, holding out Q1 2025 as an untouched evaluation set.

### 🚀 Getting Started

**1. Install Dependencies**
```bash
pip install -r atm_cash_forecast/requirements.txt
```

**2. Generate Synthetic Data**
Simulate 3 years of ATM transaction histories, including weekend bumps and geographic variations.
```bash
python atm_cash_forecast/src/data_generator.py
```

**3. Feature Engineering**
Transform raw logs into time-series features (log scaling, lags, distances to holidays).
```bash
python atm_cash_forecast/src/feature_engineering.py
```

**4. Train the Model**
You can run training directly with defaults, or use CLI overrides to tune hyperparameters without editing the `config.py` file:
```bash
python atm_cash_forecast/src/train.py --hidden_dim 64 --num_heads 4 --learning_rate 0.0005 --history_days 90
```

### 📈 Tracking with TensorBoard

All training metrics, including hyperparameter grid comparisons, are automatically logged.

To view the dashboard, run:
```bash
tensorboard --logdir atm_cash_forecast/lightning_logs
```

**Reading the Metrics:**
*   Navigate to the **Scalars** tab and search for `loss` in the **Filter tags** box.
*   Compare `val_loss` against `train_loss_epoch`.
*   *Note on the X-Axis:* TensorBoard's X-Axis measures **Global Steps** (total number of processed batches), not Epochs. For example, if you have 730 batches per epoch, step ~11,000 corresponds to epoch 15. The training ends when the validation loss flattens out and Early Stopping triggers.
*   Use the **HPARAMS** tab to compare different model configurations side-by-side to find your best combination of `hidden_dim`, `dropout`, etc.

### 📦 Training Artifacts & Model Checkpoints

During training, a `ModelCheckpoint` callback actively monitors the PyTorch Lightning `val_loss` metric. Whenever the model hits a new absolute minimum validation loss, the callback dynamically saves those "best" model weights to disk. By the time training completes (or triggers Early Stopping), the file left on disk is mathematically guaranteed to be your peak-performing architecture.

When training concludes, your optimized `.ckpt` model files can be found in:
```text
atm_cash_forecast/data/models/
```

### 🔮 Inference on Hold-Out Data

Once a model is successfully trained, use the prediction script to test its capabilities against completely unseen future data (Q1 2025). 

```bash
python atm_cash_forecast/src/predict.py
```

**What the inference script does:**
1. Automatically scans `data/models/` and selects the most recently created `.ckpt` file (the peak model from your last run).
2. Extracts the last `history_days` (e.g., 90 days) from the `train_features.csv` to use as the model's required historical context.
3. Automatically stitches that context to the initial `forecast_days` (e.g., 35 days) of the `holdout_features.csv` ground-truth target data.
4. Generates the probabilistic forecasts (the central median line + 7 surrounding Quantiles forming confidence bands).
5. Outputs highly detailed line charts plotting your specific confidence boundaries overlaid directly against what technically happened in 2025.

You can find the visually generated prediction outputs saved for 3 unique ATMs inside:
```text
atm_cash_forecast/results/
```



