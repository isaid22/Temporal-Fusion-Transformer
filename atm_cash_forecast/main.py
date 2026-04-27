# main.py
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import config
from src.data_generator import generate_training_data

def main():
    print("="*60)
    print("ATM CASH DEMAND FORECASTING SYSTEM")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  - ATMs: {config.n_atms}")
    print(f"  - Date range: {config.start_date} to {config.end_date}")
    print(f"  - History: {config.history_days} days")
    print(f"  - Forecast: {config.forecast_days} days")
    print(f"  - Denominations: {config.denominations}")
    
    # Step 1: Generate or load data
    print("\n" + "="*60)
    print("STEP 1: DATA GENERATION")
    print("="*60)
    
    if os.path.exists(f'{config.data_path}/atm_daily_operations.csv'):
        print("Loading existing data...")
        df = pd.read_csv(f'{config.data_path}/atm_daily_operations.csv')
        print(f"Loaded {len(df):,} records")
    else:
        print("Generating new synthetic data...")
        df = generate_training_data()
    
    # Display basic statistics
    print("\n" + "="*60)
    print("DATA STATISTICS")
    print("="*60)
    
    print(f"\nDate range: {df['snapshot_date'].min()} to {df['snapshot_date'].max()}")
    print(f"Total withdrawals: {df['n_withdrawals'].sum():,.0f}")
    print(f"Total deposits: {df['n_deposits'].sum():,.0f}")
    
    print("\nDenomination usage (bills):")
    for denom in config.denominations:
        withdrawn = df[f'withdrawn_{denom}'].sum()
        deposited = df[f'deposited_{denom}'].sum()
        print(f"  ${denom}: withdrawn={withdrawn:,.0f}, deposited={deposited:,.0f}, net={withdrawn-deposited:+,.0f}")
    
    print("\n✅ Setup complete! Ready for next steps.")
    print("\nNext steps:")
    print("  1. Run feature engineering: python src/feature_engineering.py")
    print("  2. Train model: python src/train.py")
    print("  3. Make predictions: python src/predict.py")

if __name__ == "__main__":
    main()