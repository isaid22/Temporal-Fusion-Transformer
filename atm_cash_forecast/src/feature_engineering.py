import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import holidays
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import config

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract temporal variables from the date"""
    print("Generating time features...")
    df['snapshot_date'] = pd.to_datetime(df['snapshot_date'])
    
    # Basic extracted dates
    df['day_of_week'] = df['snapshot_date'].dt.dayofweek
    df['day_of_month'] = df['snapshot_date'].dt.day
    df['month'] = df['snapshot_date'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Holiday calendar inclusion
    print("Assigning holidays...")
    min_year = df['snapshot_date'].dt.year.min()
    max_year = df['snapshot_date'].dt.year.max()
    us_holidays = holidays.US(years=range(min_year, max_year + 1))
    
    # Check if day is holiday or near a holiday
    df['is_holiday_str'] = df['snapshot_date'].apply(lambda x: us_holidays.get(x, 'None'))
    df['is_holiday'] = (df['is_holiday_str'] != 'None').astype(int)
    
    print("Calculating proximity to holidays...")
    df_unique_dates = pd.DataFrame({'snapshot_date': df['snapshot_date'].unique()}).sort_values('snapshot_date')
    
    # Create a DataFrame of just the holiday dates
    holidays_df = pd.DataFrame({'holiday_date': pd.to_datetime([d for d in us_holidays.keys()])}).sort_values('holiday_date')
    
    # Merge to find the next future holiday
    merged_next = pd.merge_asof(df_unique_dates, holidays_df, left_on='snapshot_date', right_on='holiday_date', direction='forward')
    df_unique_dates['days_until_next_holiday'] = (merged_next['holiday_date'] - df_unique_dates['snapshot_date']).dt.days
    
    # Merge to find the most recent past holiday
    merged_last = pd.merge_asof(df_unique_dates, holidays_df, left_on='snapshot_date', right_on='holiday_date', direction='backward')
    df_unique_dates['days_since_last_holiday'] = (df_unique_dates['snapshot_date'] - merged_last['holiday_date']).dt.days
    
    # Fill NAs for edges (e.g. if no past/future holiday is found in the truncated calendar window)
    df_unique_dates.fillna(365, inplace=True)
    
    # Map back to main DataFrame
    df = df.merge(df_unique_dates, on='snapshot_date', how='left')
    
    # Convert dates to time index per TFT requirement
    date_map = {date: i for i, date in enumerate(sorted(df['snapshot_date'].unique()))}
    df['time_idx'] = df['snapshot_date'].map(date_map)
    
    return df

def add_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate lag features and rolling averages based on historical data"""
    print("Generating lag and rolling window features...")
    
    # Need to sort by atm_id and snapshot_date
    df = df.sort_values(by=['atm_id', 'snapshot_date']).reset_index(drop=True)
    
    # The target to forecast is typically the total withdrawal amount or net cash flow
    target_col = 'withdrawal_amount_total'
    
    # Lags (past observations)
    for lag in [1, 7, 14, 28]:
        df[f'{target_col}_lag_{lag}'] = df.groupby('atm_id')[target_col].shift(lag)
        
    # Rolling averages (using lag to avoid data leakage)
    for window in [7, 14, 30]:
        roll_values = df.groupby('atm_id')[f'{target_col}_lag_1'].rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
        df[f'{target_col}_rolling_mean_{window}'] = roll_values
    
    # Fill NAs created by shift/rolling with 0 or mean
    df.fillna(0, inplace=True)
    
    return df

def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize features (Neural networks prefer scaled values)"""
    print("Scaling continuous features...")
    # Use max-abs or standard scaler; here we'll do log1p for vast absolute values like dollars
    continuous_cols = [c for c in df.columns if ('amount' in c or 'lag' in c or 'rolling' in c or 'closing' in c)]
    
    for col in continuous_cols:
        df[f"{col}_scaled"] = np.log1p(df[col].astype(float))
        
    return df

def run_fe_pipeline():
    print("="*60)
    print("FEATURE ENGINEERING PIPELINE")
    print("="*60)
    
    # 1. Process Training Data
    print("\nProcessing Training Data...")
    train_path = f'{config.data_path}/atm_daily_operations.csv'
    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found. Run generator first.")
        return
        
    df_train = pd.read_csv(train_path)
    df_train = add_time_features(df_train)
    df_train = add_lag_and_rolling_features(df_train)
    df_train = scale_features(df_train)
    
    # Save engineered training data
    out_train = f'{config.data_path}/train_features.csv'
    df_train.to_csv(out_train, index=False)
    print(f"✅ Saved engineered training data to {out_train} (Shape: {df_train.shape})")

    # 2. Process Holdout Data
    print("\nProcessing Holdout Data...")
    holdout_path = f'{config.data_path}/atm_holdout_operations.csv'
    if os.path.exists(holdout_path):
        df_holdout = pd.read_csv(holdout_path)
        df_holdout = add_time_features(df_holdout)
        df_holdout = add_lag_and_rolling_features(df_holdout)
        df_holdout = scale_features(df_holdout)
        
        out_holdout = f'{config.data_path}/holdout_features.csv'
        df_holdout.to_csv(out_holdout, index=False)
        print(f"✅ Saved engineered holdout data to {out_holdout} (Shape: {df_holdout.shape})")

if __name__ == "__main__":
    run_fe_pipeline()
