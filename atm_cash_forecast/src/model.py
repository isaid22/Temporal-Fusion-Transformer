import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet

def build_dataset(df: pd.DataFrame, max_encoder_length: int = 60, max_prediction_length: int = 35) -> TimeSeriesDataSet:
    """
    Builds the PyTorch Forecasting TimeSeriesDataSet.
    This tells the TFT model exactly what each column in our dataframe represents.
    """
    # We want to use all available history up to max_encoder_length
    # to predict exactly max_prediction_length into the future
    
    training = TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="withdrawal_amount_total",
        group_ids=["atm_id"],
        min_encoder_length=max_encoder_length // 2,  # keep encoder length flexible
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        
        # Static covariates (these do NOT change over time for a given ATM)
        static_categoricals=["atm_id", "city"],
        static_reals=["latitude", "longitude"],
        
        # Time-varying known covariates (these change over time, but we KNOW them in the future)
        time_varying_known_categoricals=["day_of_week", "month"],
        time_varying_known_reals=["time_idx", "days_until_next_holiday", "is_weekend", "is_holiday"],
        
        # Time-varying unknown covariates (these change, but we DO NOT know them in the future)
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=[
            "withdrawal_amount_total",
            "deposit_amount_total_scaled",
            "n_withdrawals",
            "n_deposits",
            "withdrawal_amount_total_lag_1_scaled",
            "withdrawal_amount_total_lag_7_scaled",
            "withdrawal_amount_total_rolling_mean_7_scaled",
            "days_since_last_holiday"
        ],
        target_normalizer=None,  # We pre-scaled our target log1p, or we could use GroupNormalizer
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    return training
