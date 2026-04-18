import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from pathlib import Path
import logging
from compspoofv2_dataset import create_compspoofv2_dataloader

# Import your exact original function
# from your_dataset_file import create_compspoofv2_dataloader

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. SETUP DATALOADERS (Outside the Trial)
# ==========================================
# CRITICAL: We initialize the datasets OUTSIDE the objective function.
# This ensures that `cache_spectrograms=True` builds the RAM cache during Trial 1,
# and all subsequent trials (Trial 2 to 100) just read instantly from RAM.

logging.info("Initializing DataLoaders and building cache...")

train_loader, train_dataset = create_compspoofv2_dataloader(
    csv_path="CompSpoofV2/development/metadata/train.csv",
    dataset_root="CompSpoofV2",
    batch_size=32, # Keeping batch size fixed makes caching easier
    shuffle=True,
    num_workers=0, # Set to 0 if caching to RAM to avoid multiprocessing memory duplication
    cache_spectrograms=True 
)

val_loader, val_dataset = create_compspoofv2_dataloader(
    csv_path="CompSpoofV2/development/metadata/val.csv", # Assuming you made a val split
    dataset_root="CompSpoofV2",
    batch_size=32,
    shuffle=False,
    num_workers=0,
    cache_spectrograms=True
)

# Optional: Get your class weights for the loss function
class_weights = train_dataset.get_class_weights().to(device)


# ==========================================
# 2. THE OPTUNA OBJECTIVE FUNCTION
# ==========================================
def objective(trial):
    # --- A. Hyperparameters to Tune ---
    # Optuna will pick these values intelligently each trial
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(64 * 300, 128), # Assuming (Batch, 1, 64, ~300)
        nn.ReLU(),
        nn.Dropout(trial.suggest_float("dropout", 0.2, 0.5)),
        nn.Linear(128, 5) # 5 classes in your label_mapping
    ).to(device)
    
    # --- C. Optimizer & Loss ---
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # --- D. Training Loop ---
    epochs = 10 # Keep epochs relatively low for hyperparameter searches
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            # Your dataloader returns a dict with these exact keys
            inputs = batch['spectrogram'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        # --- E. Validation Loop ---
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['spectrogram'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        val_accuracy = correct / total
        
        # --- F. Optuna Pruning (The Magic Feature) ---
        # Report the metric to Optuna
        trial.report(val_accuracy, epoch)
        
        # If the trial is performing terribly compared to previous trials, kill it early
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    return val_accuracy

# ==========================================
# 3. RUNNING THE STUDY
# ==========================================
if __name__ == "__main__":
    # Create a study object that aims to maximize our validation accuracy
    # MedianPruner kills unpromising trials early to save compute time
    study = optuna.create_study(
        direction="maximize", 
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    )
    
    logging.info("Starting Optuna Hyperparameter Optimization...")
    study.optimize(objective, n_trials=50) # Run 50 different hyperparameter combinations
    
    # Print the results
    print("\nStudy complete!")
    print("Best Trial:")
    trial = study.best_trial
    
    print(f"  Value (Val Accuracy): {trial.value:.4f}")
    print("  Best Hyperparameters:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")