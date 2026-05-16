# Temporal-Fusion-Transformer
This is a repo for multiple examples of time series problems using Temporal Fusion Transformers.

## Project 1: ATM Cash Demand Forecasting

This project demonstrates an end-to-end framework for forecasting multi-horizon daily cash demand across a simulated network of 100 ATMs distributed across 4 major US cities. By leveraging the Temporal Fusion Transformer (TFT), the model identifies complex time-varying relationships (like payday cycles and holiday spikes) alongside static infrastructure covariates (like ATM location).

## Documentation Index

- [README.md](README.md): Main project overview, training, inference, and evaluation workflow.
- [atm_cash_forecast/README_TUNING.md](atm_cash_forecast/README_TUNING.md): Hyperparameter tuning instructions, search-space config usage, and results locations.
- [atm_cash_forecast/README_TENSORBOARD.md](atm_cash_forecast/README_TENSORBOARD.md): TensorBoard usage for both training logs and Ray Tune results.
- [atm_cash_forecast/data/data-reference.md](atm_cash_forecast/data/data-reference.md): Data dictionary and field-level reference notes.

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

### 🔮 Inference & Evaluation

Once a model is successfully trained, use the prediction script to test its capabilities against completely unseen future data (Q1 2025).

```bash
python atm_cash_forecast/src/predict.py
```

The script automatically loads your absolute best `.ckpt` model, extracts the necessary 90-day historical context, and generates a fully probabilistic 35-day forecast. All outputs are saved to `atm_cash_forecast/results/`.

#### 1. Visual Forecast Plots (`forecast_atm_*.png`)
These plots visually overlay the model's forecasting confidence against the actual ground-truth data.

*   **The Grey Line (Actuals):** Represents the true historical data (past) and the actual ground truth of the holdout period (future).
*   **The Solid Orange Line (Median):** The model's 50th quantile (its primary point-forecast).
*   **The Orange Shaded Bands (Confidence Intervals):**
    *   **Darkest Orange:** The 25th–75th quantile band (50% confidence).
    *   **Medium Orange:** The 10th–90th quantile band (80% confidence).
    *   **Lightest Faint Orange:** The 2nd–98th quantile absolute extremes. If the grey line spikes outside this faint cloud, a severe anomaly occurred that broke the model's highest boundaries.
    *   *Interpretation:* A narrow orange band means the model is highly confident. A wide orange band means the model recognizes high uncertainty (e.g., holidays/weekends).
*   **The Bottom Grey Line (Attention):** Found spanning the past (negative X-axis), this shows exactly which historical days the Multi-Head Attention mechanism found most useful to generate the future prediction (e.g., spikes indicating weekly seasonality).

#### 2. Aggregated Metrics (`holdout_evaluation_metrics.json`)
Evaluates the **50th Quantile (Median)** prediction against the actual ground truth across all 100 ATMs. *(Note: The unit of measurement is 'number of bills', not dollar amounts)*.

*   **MAE (Mean Absolute Error):** On average, how many bills the forecast was off by per day.
*   **RMSE (Root Mean Squared Error):** Penalizes larger errors more heavily. Useful to gauge if the model has severe misses.
*   **sMAPE (Symmetric Mean Absolute Percentage Error):** The percentage error. sMAPE is used instead of standard MAPE because daily cash demand can sometimes drop near zero, which would cause division-by-zero crashes.

#### 3. Raw Probabilistic Data (`raw_predictions.csv`)
Because the TFT outputs a full probability distribution, you get every single measurement day-by-day and ATM-by-ATM in a flat CSV.

*   **Columns:** `atm_id`, `time_idx`, `actual_bills`, `absolute_error`, and all 7 predicted quantiles (`predicted_q0.02` ... `predicted_q0.98`).
*   **Business Use Case:** Instead of stocking the exact median amount (`predicted_q0.50`), ATM operators can simply pull the `predicted_q0.90` (the 90th percentile) column to ensure the machine has enough cash 90% of the time, safely mathematically preventing stock-outs without guessing!



