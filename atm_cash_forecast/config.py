# config.py
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Config:
    # Data parameters
    n_atms: int = 100
    start_date: str = '2022-01-01'
    end_date: str = '2024-12-31'
    holdout_start_date: str = '2025-01-01'
    holdout_end_date: str = '2025-02-28'
    history_days: int = 90
    forecast_days: int = 35
    
    # Model parameters
    hidden_dim: int = 256
    num_heads: int = 8
    dropout: float = 0.1
    embedding_dim: int = 32
    statistical_feature_dim: int = 50
    
    # Training parameters
    batch_size: int = 128
    learning_rate: float = 0.001
    epochs: int = 20
    early_stopping_patience: int = 10
    
    # Denomination configuration
    denominations: list = None  # Will be set in __post_init__
    
    # Paths
    data_path: str = 'data/'
    model_save_path: str = 'data/models/'
    
    def __post_init__(self):
        if self.denominations is None:
            self.denominations = [5, 20, 50, 100]

config = Config()