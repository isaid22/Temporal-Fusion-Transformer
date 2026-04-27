# src/data_generator.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum
import random
from tqdm import tqdm
import sys
import os
import holidays
    
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import config

class Denomination(Enum):
    FIVE = 5
    TWENTY = 20
    FIFTY = 50
    HUNDRED = 100

@dataclass
class Cassette:
    denomination: Denomination
    capacity: int
    current_count: int
    min_threshold: int
    max_threshold: int

class ATMSimulator:
    """Simulates ATM operations with realistic patterns"""
    
    def __init__(self):
        self.denominations = [Denomination.FIVE, Denomination.TWENTY, 
                              Denomination.FIFTY, Denomination.HUNDRED]
        
        # Withdrawal amount distributions (in dollars)
        self.withdrawal_amounts = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 
                                   240, 280, 300, 400, 500]
        self.withdrawal_weights = [0.15, 0.12, 0.10, 0.08, 0.15, 0.08, 0.05, 0.05, 
                                   0.08, 0.05, 0.03, 0.02, 0.02, 0.01, 0.01]
        
        # City centers for location generation (Lat, Lon)
        self.city_centers = {
            'New York City, NY': (40.7484, -73.9857),  # Midtown/Lower Manhattan focus
            'Los Angeles, CA': (34.0522, -118.2437),
            'Dallas, TX': (32.7767, -96.7970),
            'Chicago, IL': (41.8781, -87.6298)
        }
        self.cities = list(self.city_centers.keys())
        self.us_holidays = holidays.US(years=range(2021, 2027))
        
    def generate_atm_behavior_profile(self, atm_id: int) -> Dict:
        """
        Generate unique behavior profile for each ATM with geographical locations
        """
        # Assign 25 ATMs to each city
        city_idx = (atm_id % 100) // 25
        city_name = self.cities[city_idx]
        center_lat, center_lon = self.city_centers[city_name]
        
        # Add random noise within approx ~1-2 miles (+/- 0.02 degrees)
        lat = center_lat + np.random.uniform(-0.02, 0.02)
        lon = center_lon + np.random.uniform(-0.02, 0.02)
        
        # Random but realistic behavior patterns
        return {
            'atm_id': atm_id,
            'city': city_name,
            'latitude': round(lat, 6),
            'longitude': round(lon, 6),
            'prefers_20s': np.random.uniform(0.4, 0.8),  # 20s preference
            'prefers_50s': np.random.uniform(0.1, 0.4),  # 50s preference
            'prefers_100s': np.random.uniform(0.05, 0.3), # 100s preference
            'weekend_boost': np.random.uniform(1.1, 1.6), # Weekend volume increase
            'holiday_boost': np.random.uniform(1.3, 2.0), # Holiday volume increase
            'deposit_likelihood': np.random.uniform(0.1, 0.4), # Chance a transaction is deposit
            'avg_withdrawal': np.random.choice([60, 80, 100, 120, 150, 200]),
            'volatility': np.random.uniform(0.1, 0.3),  # Day-to-day variation
        }
    
    def _get_bill_mix(self, amount: int, profile: Dict) -> Dict[Denomination, int]:
        """Determine which bills to dispense based on ATM's profile"""
        bills = {d: 0 for d in self.denominations}
        remaining = amount
        
        # Use $20s first (most common)
        n_twenties = min(remaining // 20, int(profile['prefers_20s'] * 15))
        if n_twenties > 0:
            bills[Denomination.TWENTY] = n_twenties
            remaining -= n_twenties * 20
        
        # Then $50s
        if remaining >= 50:
            n_fifties = min(remaining // 50, int(profile['prefers_50s'] * 8))
            if n_fifties > 0:
                bills[Denomination.FIFTY] = n_fifties
                remaining -= n_fifties * 50
        
        # Then $100s
        if remaining >= 100:
            n_hundreds = min(remaining // 100, int(profile['prefers_100s'] * 5))
            if n_hundreds > 0:
                bills[Denomination.HUNDRED] = n_hundreds
                remaining -= n_hundreds * 100
        
        # Mop up with $20s if needed
        if remaining > 0:
            extra_20s = remaining // 20
            if extra_20s > 0:
                bills[Denomination.TWENTY] += extra_20s
        
        return bills
    
    def simulate_day(self, atm_id: int, profile: Dict, date: datetime, 
                    current_inventory: Dict[Denomination, int]) -> Dict:
        """Simulate one day of ATM operations"""
        
        # Determine transaction volume
        base_volume = profile['avg_withdrawal'] / 20  # Rough conversion
        day_of_week = date.weekday()
        
        # Day of week factor
        dow_factors = [0.85, 0.85, 0.85, 0.9, 1.1, 1.4, 1.2]
        volume = int(base_volume * dow_factors[day_of_week])
        
        # Weekend boost
        if day_of_week >= 5:
            volume = int(volume * profile['weekend_boost'])
            
        # Holiday check (exact day or day before)
        is_holiday = date in self.us_holidays
        is_holiday_eve = (date + timedelta(days=1)) in self.us_holidays
        
        if is_holiday or is_holiday_eve:
            volume = int(volume * profile['holiday_boost'])
        
        # Add randomness
        volume = int(volume * np.random.normal(1.0, profile['volatility']))
        volume = max(5, min(200, volume))
        
        # Track daily totals
        withdrawals = {d: 0 for d in self.denominations}
        deposits = {d: 0 for d in self.denominations}
        withdrawal_amount_total = 0
        deposit_amount_total = 0
        n_withdrawals = 0
        n_deposits = 0
        
        # Process each transaction
        for _ in range(volume):
            is_deposit = np.random.random() < profile['deposit_likelihood']
            
            if is_deposit:
                # Deposit transaction
                amount = np.random.choice([50, 100, 200, 300, 500, 1000])
                deposit_amount_total += amount
                n_deposits += 1
                
                # For deposits, assume $20s are most common
                deposit_bills = {Denomination.TWENTY: amount // 20}
                if amount % 20 != 0:
                    deposit_bills[Denomination.FIVE] = 1
                
                for denom, count in deposit_bills.items():
                    deposits[denom] += count
                    if denom in current_inventory:
                        current_inventory[denom] += count
            else:
                # Withdrawal transaction
                amount = np.random.choice(self.withdrawal_amounts, p=self.withdrawal_weights)
                withdrawal_amount_total += amount
                n_withdrawals += 1
                
                # Get bill mix based on ATM profile
                bill_mix = self._get_bill_mix(amount, profile)
                
                # Check if enough bills available
                can_dispense = all(current_inventory.get(d, 0) >= bill_mix.get(d, 0) 
                                  for d in self.denominations)
                
                if can_dispense:
                    for denom, count in bill_mix.items():
                        withdrawals[denom] += count
                        current_inventory[denom] -= count
                else:
                    # Fallback: dispense what's available
                    pass
        
        # Check if replenishment needed
        was_refilled = False
        refill_amounts = {d: 0 for d in self.denominations}
        
        for denom in self.denominations:
            # Reorder at 20% of initial capacity
            initial_capacity = 4000 if denom == Denomination.TWENTY else 2500
            if current_inventory[denom] < initial_capacity * 0.2:
                was_refilled = True
                # Refill to 80%
                target = int(initial_capacity * 0.8)
                refill_amounts[denom] = max(0, target - current_inventory[denom])
                current_inventory[denom] += refill_amounts[denom]
        
        return {
            # === TRANSACTION METRICS (counts) ===
            'snapshot_date': date,                           # DATE: Calendar date
            'atm_id': atm_id,                                # INTEGER: ATM identifier
            'city': profile['city'],                         # STRING: City name
            'latitude': profile['latitude'],                 # FLOAT: Geo Latitude
            'longitude': profile['longitude'],               # FLOAT: Geo Longitude
            
            'n_withdrawals': n_withdrawals,                  # COUNT: Number of withdrawal transactions
            'n_deposits': n_deposits,                        # COUNT: Number of deposit transactions
            
            # === MONETARY VALUES (dollars) ===
            'withdrawal_amount_total': withdrawal_amount_total,  # DOLLARS: Total cash withdrawn
            'deposit_amount_total': deposit_amount_total,        # DOLLARS: Total cash deposited
            
            # === BILL COUNTS (physical bills) ===
            # WITHDRAWALS (bills dispensed to customers)
            'withdrawn_5': withdrawals.get(Denomination.FIVE, 0),     # COUNT: $5 bills dispensed
            'withdrawn_20': withdrawals.get(Denomination.TWENTY, 0),  # COUNT: $20 bills dispensed
            'withdrawn_50': withdrawals.get(Denomination.FIFTY, 0),   # COUNT: $50 bills dispensed
            'withdrawn_100': withdrawals.get(Denomination.HUNDRED, 0), # COUNT: $100 bills dispensed
            
            # DEPOSITS (bills received from customers)
            'deposited_5': deposits.get(Denomination.FIVE, 0),     # COUNT: $5 bills deposited
            'deposited_20': deposits.get(Denomination.TWENTY, 0),  # COUNT: $20 bills deposited
            'deposited_50': deposits.get(Denomination.FIFTY, 0),   # COUNT: $50 bills deposited
            'deposited_100': deposits.get(Denomination.HUNDRED, 0), # COUNT: $100 bills deposited
            
            # REPLENISHMENT (bills added by vendor)
            'refill_5': refill_amounts.get(Denomination.FIVE, 0),     # COUNT: $5 bills added
            'refill_20': refill_amounts.get(Denomination.TWENTY, 0),  # COUNT: $20 bills added
            'refill_50': refill_amounts.get(Denomination.FIFTY, 0),   # COUNT: $50 bills added
            'refill_100': refill_amounts.get(Denomination.HUNDRED, 0), # COUNT: $100 bills added
            
            # === OPERATIONAL FLAGS ===
            'was_refilled': was_refilled,                      # BOOLEAN: Whether ATM was refilled
            
            # === INVENTORY (bills remaining in ATM) ===
            'closing_5': current_inventory.get(Denomination.FIVE, 0),     # COUNT: $5 bills at day end
            'closing_20': current_inventory.get(Denomination.TWENTY, 0),  # COUNT: $20 bills at day end
            'closing_50': current_inventory.get(Denomination.FIFTY, 0),   # COUNT: $50 bills at day end
            'closing_100': current_inventory.get(Denomination.HUNDRED, 0), # COUNT: $100 bills at day end
        }

def generate_training_data():
    """Generate synthetic training data for all ATMs"""
    
    print("="*60)
    print("GENERATING SYNTHETIC ATM DATA")
    print("="*60)
    
    simulator = ATMSimulator()
    profiles = {}
    
    # Generate profiles for each ATM
    for atm_id in range(config.n_atms):
        profiles[atm_id] = simulator.generate_atm_behavior_profile(atm_id)
        
    # Create and save location lookup table
    location_data = []
    for atm_id, profile in profiles.items():
        location_data.append({
            'atm_id': profile['atm_id'],
            'city': profile['city'],
            'latitude': profile['latitude'],
            'longitude': profile['longitude']
        })
    loc_df = pd.DataFrame(location_data)
    os.makedirs(config.data_path, exist_ok=True)
    loc_df.to_csv(f'{config.data_path}/atm_locations.csv', index=False)
    print(f"✅ Location lookup table saved: {config.data_path}/atm_locations.csv")
    
    # Generate daily data
    all_data = []
    date_range = pd.date_range(config.start_date, config.end_date, freq='D')
    
    # Initialize inventory for each ATM
    inventories = {}
    for atm_id in range(config.n_atms):
        inventories[atm_id] = {
            Denomination.FIVE: 2000,
            Denomination.TWENTY: 4000,
            Denomination.FIFTY: 2500,
            Denomination.HUNDRED: 2000,
        }
    
    print(f"\nSimulating {len(date_range)} days for {config.n_atms} ATMs...")
    
    for date in tqdm(date_range, desc="Processing days"):
        for atm_id in range(config.n_atms):
            result = simulator.simulate_day(
                atm_id, profiles[atm_id], date, inventories[atm_id]
            )
            all_data.append(result)
            
    # Generate holdout data
    holdout_data = []
    holdout_date_range = pd.date_range(config.holdout_start_date, config.holdout_end_date, freq='D')
    
    print(f"\nSimulating {len(holdout_date_range)} holdout days for {config.n_atms} ATMs...")
    for date in tqdm(holdout_date_range, desc="Processing holdout days"):
        for atm_id in range(config.n_atms):
            result = simulator.simulate_day(
                atm_id, profiles[atm_id], date, inventories[atm_id]
            )
            holdout_data.append(result)
    
    df = pd.DataFrame(all_data)
    holdout_df = pd.DataFrame(holdout_data)
    
    # Save to CSV
    os.makedirs(config.data_path, exist_ok=True)
    df.to_csv(f'{config.data_path}/atm_daily_operations.csv', index=False)
    holdout_df.to_csv(f'{config.data_path}/atm_holdout_operations.csv', index=False)
    
    print(f"\n✅ Data generated: {len(df):,} records")
    print(f"✅ Holdout Data generated: {len(holdout_df):,} records")
    print(f"   Saved to: {config.data_path}/atm_daily_operations.csv")
    print(f"   Saved to: {config.data_path}/atm_holdout_operations.csv")
    
    return df

if __name__ == "__main__":
    df = generate_training_data()
    print("\nSample data:")
    print(df.head(10))