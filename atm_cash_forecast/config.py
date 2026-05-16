# config.py
import os
import yaml
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Config:
    # Data parameters
    n_atms: int
    start_date: str
    end_date: str
    holdout_start_date: str
    holdout_end_date: str
    history_days: int
    forecast_days: int
    
    # Model parameters
    hidden_dim: int
    num_heads: int
    dropout: float
    embedding_dim: int
    
    # Training parameters
    batch_size: int
    learning_rate: float
    epochs: int
    early_stopping_patience: int
    
    # Denomination configuration
    denominations: Optional[List[int]] = None
    
    # Paths
    data_path: str = 'data/'
    model_save_path: str = 'data/models/'
    
    def __post_init__(self):
        if self.denominations is None:
            self.denominations = [5, 20, 50, 100]

def load_config(config_file="config.yaml") -> Config:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(base_dir, config_file)
    with open(yaml_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return Config(**config_dict)

config = load_config()