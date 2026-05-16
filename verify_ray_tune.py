import os
import torch
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

# 1. Silence underlying environment warnings
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"

def train_func(config):
    # Verify the worker process has clean access to your RTX 4060
    if torch.cuda.is_available():
        print(f"\n[WORKER] Success! Training on hardware: {torch.cuda.get_device_name(0)}\n")
    else:
        print("\n[WORKER] Error: Worker could not see the GPU!\n")
    
    # Simple mathematical objective function to optimize
    loss = (config["alpha"] - 2) ** 2
    
    # Report the metric back to the scheduler
    tune.report({"loss": loss})

# Initialize local ray orchestration engine
ray.init(ignore_reinit_error=True)

# Define search parameter range
search_space = {"alpha": tune.uniform(0, 5)}

# 2. Configure Bayesian Search (Optuna) & Early Stopping (ASHA / Hyperband)
optuna_search = OptunaSearch(metric="loss", mode="min")
asha_scheduler = ASHAScheduler(metric="loss", mode="min", max_t=10, grace_period=1)

# Bind exactly 1 GPU resource explicitly to the training workflow
trainable_with_gpu = tune.with_resources(train_func, resources={"gpu": 1})

# 3. FIX: Execute via tune.run to bypass the buggy experimental Tuner class
analysis = tune.run(
    trainable_with_gpu,
    num_samples=4,               # Number of unique trials to explore
    search_alg=optuna_search,    # Apply Bayesian Optimization
    scheduler=asha_scheduler,    # Apply Early-Stopping Hyperband
    config=search_space,
    verbose=1                    # Forces primitive integer flag to avoid string parsing crashes
)

print("\n==============================")
print("  VERIFICATION SUCCESSFUL!   ")
print(f"Best Configuration Found: {analysis.get_best_config(metric='loss', mode='min')}")
print("==============================\n")
