# %%
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm  # For progress bars
import time

# Configuration
DATA_FILE = 'data_clean.csv'
OUTPUT_PREDICTIONS_FILE = 'predictions.csv'
FORECAST_HORIZON = 240  # 4 hours in minutes
N_SPLITS_CV = 5         # Number of splits for TimeSeriesSplit
OPTUNA_TRIALS = 30      # Number of Optuna trials (adjust as needed)
TRAIN_SPLIT_RATIO = 0.9 # Use 90% for training, 10% for testing

# --- 1. Load Data ---
print("Loading data...")
data = pd.read_csv(DATA_FILE, parse_dates=['Date'], index_col='Date')
# Ensure data has a frequency, necessary for shifts and rolling windows
# Infer frequency if possible, otherwise set explicitly if known (e.g., 'T' for minutes)
inferred_freq = pd.infer_freq(data.index)
if inferred_freq:
    # Fix: Ensure freq string is standard (e.g., 'T' or 'min' -> 'T')
    if 'min' in inferred_freq.lower():
         inferred_freq = 'T'
    print(f"Inferred frequency: {inferred_freq}")
    data = data.asfreq(inferred_freq)
else:
    print("Could not infer frequency. Assuming minute frequency ('T'). Check data index if errors occur.")
    # Resample might be needed if there are gaps despite interpolation
    data = data.asfreq('T')
    # Re-interpolate after setting frequency if necessary
    data = data.interpolate(method='time')

original_cols = [col for col in data.columns if col.startswith('Series')]
print(f"Original columns: {original_cols}")

# --- 2. Feature Engineering ---
print("Starting feature engineering...")
df = data.copy()

# Calendar features
df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek
df['day_of_year'] = df.index.dayofyear
df['month'] = df.index.month
# Fix: Ensure week of year is treated as integer
df['week_of_year'] = df.index.isocalendar().week.astype(int)
df['year'] = df.index.year
df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)

# Lagged features (self and cross-series)
lags_self = [1, 5, 15, 60, FORECAST_HORIZON, FORECAST_HORIZON + 15] # Lags for the series itself
lags_cross = [1, 5, FORECAST_HORIZON] # Lags for other series impacting the target series

# Store new feature names
new_feature_cols = []

# Calendar features added first
new_feature_cols.extend(['hour', 'day_of_week', 'day_of_year', 'month', 'week_of_year', 'year', 'is_weekend'])


for col in tqdm(original_cols, desc="Generating Lag Features"):
    # Self lags
    for lag in lags_self:
        feature_name = f'{col}_lag_{lag}'
        df[feature_name] = df[col].shift(lag)
        new_feature_cols.append(feature_name)
    # Cross-series lags (feature for predicting `col`)
    for other_col in original_cols:
        if col == other_col:
            continue
        for lag in lags_cross:
             feature_name = f'{col}_X_{other_col}_lag_{lag}'
             df[feature_name] = df[other_col].shift(lag)
             new_feature_cols.append(feature_name)


# Rolling window features (self only for now, add cross-series if needed)
windows = [15, 60, FORECAST_HORIZON] # e.g., 15min, 1hr, 4hr windows

for col in tqdm(original_cols, desc="Generating Rolling Features"):
    for w in windows:
        # Use closed='left' to prevent data leakage from current time step for rolling calculations
        rolling_obj = df[col].rolling(window=w, min_periods=3, closed='left') # min_periods helps with NaNs at start

        mean_feat_name = f'{col}_rollmean_{w}'
        std_feat_name = f'{col}_rollstd_{w}'

        df[mean_feat_name] = rolling_obj.mean()
        df[std_feat_name] = rolling_obj.std()

        new_feature_cols.extend([mean_feat_name, std_feat_name])


print(f"Total features generated: {len(new_feature_cols)}")


# Fix: De-fragment DataFrame after adding columns
print("De-fragmenting DataFrame...")
df = df.copy()


# --- 3. Target Creation ---
print("Creating target variables...")
target_cols = []
for col in original_cols:
    target_col = f'{col}_target'
    df[target_col] = df[col].shift(-FORECAST_HORIZON)
    target_cols.append(target_col)

# --- 4. Handle NaNs & Split Data ---
# Drop rows with NaNs created by feature engineering (lags/rolling) or target shifting
print(f"Shape before dropping NaNs: {df.shape}")
df.dropna(inplace=True)
print(f"Shape after dropping NaNs: {df.shape}")

# Check if DataFrame is empty after dropna
if df.empty:
    raise ValueError("DataFrame is empty after dropping NaNs. Check lag/window sizes and data length.")


# Time-based split
n = int(len(df) * TRAIN_SPLIT_RATIO)
train_df = df.iloc[:n]
test_df = df.iloc[n:]

# Make sure test_df is not empty
if test_df.empty:
     raise ValueError("Test set is empty after split. Adjust TRAIN_SPLIT_RATIO or check data.")


X_train = train_df[new_feature_cols]
X_test = test_df[new_feature_cols]

# Prepare multiple targets
y_train = train_df[target_cols]
y_test = test_df[target_cols]

print(f"Train shape: X={X_train.shape}, y={y_train.shape}")
print(f"Test shape: X={X_test.shape}, y={y_test.shape}")

# --- 5. Hyperparameter Optimization (Optuna for one series) ---
print("Starting hyperparameter optimization (Optuna)...")

# Using TimeSeriesSplit for cross-validation
tscv = TimeSeriesSplit(n_splits=N_SPLITS_CV, gap=0, test_size=FORECAST_HORIZON * 2) # Test on approx 2 prediction horizons

# Target series to optimize for (e.g., Series5)
tuning_target_col = 'Series5_target' # Adjust if needed
print(f"Tuning hyperparameters based on: {tuning_target_col}")
y_train_tuning = y_train[tuning_target_col]

def objective(trial):
    params = {
        'objective': 'regression_l1', # MAE objective, often more robust to outliers than L2 (RMSE)
        'metric': 'rmse',             # Still evaluate with RMSE
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'n_jobs': -1, # Use all available cores
        'seed': 42,
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 16, 256),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 200),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0), # Colsample_bytree
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0), # Subsample
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True), # L1 regularization
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True), # L2 regularization
    }
    cv_rmses = []
    # TimeSeriesSplit requires numpy arrays potentially
    X_train_np = X_train.values
    y_train_tuning_np = y_train_tuning.values

    for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_train_np)):
        print(f"  Optuna Fold {fold+1}/{N_SPLITS_CV}")
        # Ensure indices are valid
        if len(va_idx) == 0:
            print(f"  Skipping Fold {fold+1} due to empty validation set.")
            continue

        X_tr, X_va = X_train_np[tr_idx], X_train_np[va_idx]
        y_tr, y_va = y_train_tuning_np[tr_idx], y_train_tuning_np[va_idx]

        # Check if validation set is still empty after slicing (shouldn't happen if len(va_idx)>0)
        if X_va.shape[0] == 0 or y_va.shape[0] == 0:
             print(f"  Skipping Fold {fold+1} due to empty validation arrays after slicing.")
             continue


        dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=new_feature_cols) # Use feature names here
        dvalid = lgb.Dataset(X_va, label=y_va, reference=dtrain, feature_name=new_feature_cols)

        gbm = lgb.train(
            params,
            dtrain,
            num_boost_round=1000, # Max rounds, will be stopped early
            valid_sets=[dtrain, dvalid],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0) # Suppress fold logs
            ]
        )
        preds = gbm.predict(X_va)

        # Fix: Calculate RMSE manually
        mse = mean_squared_error(y_va, preds)
        rmse = np.sqrt(mse)

        cv_rmses.append(rmse)

    # Avoid error if all folds were skipped
    if not cv_rmses:
         print("  Warning: All CV folds resulted in empty validation sets. Returning high error.")
         return float('inf')


    mean_cv_rmse = np.mean(cv_rmses)
    print(f"  Trial Mean CV RMSE: {mean_cv_rmse:.4f}")
    return mean_cv_rmse

study = optuna.create_study(direction='minimize')
start_optuna = time.time()
study.optimize(objective, n_trials=OPTUNA_TRIALS)
end_optuna = time.time()

print(f"Optuna finished in {(end_optuna - start_optuna)/60:.2f} minutes.")
print("🎯 Best Params:", study.best_params)
print(f"🎯 Best CV RMSE ({tuning_target_col}): {study.best_value:.4f}")

# --- 6. Final Model Training ---
print("Training final models for each series...")
best_params = study.best_params
# Update fixed params
best_params.update({
    'objective': 'regression_l1',
    'metric': 'rmse',
    'verbosity': -1,
    'n_jobs': -1,
    'seed': 42
    })

final_models = {}
start_train = time.time()

# Determine optimal boosting rounds using the tuning target and full training data with early stopping
print(f"Finding optimal boost rounds using {tuning_target_col}...")
# Use a validation split from the end of the training set for early stopping determination
val_size = int(len(X_train) * 0.1) # Use last 10% of train for validation here

# Ensure val_size doesn't make train part empty
if val_size >= len(X_train):
     val_size = max(1, int(len(X_train) * 0.05)) # Reduce validation size if needed
     print(f"Reduced validation size for early stopping determination to: {val_size}")

X_train_part = X_train[:-val_size]
y_train_tuning_part = y_train_tuning[:-val_size]
X_val_part = X_train[-val_size:]
y_val_tuning_part = y_train_tuning[-val_size:]


# Check if validation parts are empty
if X_val_part.empty or y_val_tuning_part.empty:
     print("Warning: Validation set for determining boost rounds is empty. Using fixed 100 rounds.")
     optimal_boost_rounds = 100
else:
    dtrain_opt = lgb.Dataset(X_train_part, label=y_train_tuning_part, feature_name=new_feature_cols)
    dvalid_opt = lgb.Dataset(X_val_part, label=y_val_tuning_part, reference=dtrain_opt, feature_name=new_feature_cols)

    temp_gbm = lgb.train(
        best_params,
        dtrain_opt,
        num_boost_round=2000, # High number, will stop early
        valid_sets=[dtrain_opt, dvalid_opt],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=True), # More patient early stopping
            lgb.log_evaluation(period=100)
        ]
    )
    optimal_boost_rounds = temp_gbm.best_iteration
    print(f"Optimal boosting rounds determined: {optimal_boost_rounds}")
    # Use a slightly higher number for final training, or the exact number. Let's use exact.
    if optimal_boost_rounds <= 0:
        print("Warning: Optimal boost rounds is 0 or negative. Using default 100.")
        optimal_boost_rounds = 100


for target_col in tqdm(target_cols, desc="Training Final Models"):
    series_y_train = y_train[target_col]

    dtrain_full = lgb.Dataset(X_train, label=series_y_train, feature_name=new_feature_cols)

    final_gbm = lgb.train(
        best_params,
        dtrain_full,
        num_boost_round=optimal_boost_rounds # Train for optimal rounds on full data
        # No validation set here as we train on all train data
    )
    final_models[target_col] = final_gbm

end_train = time.time()
print(f"Final models trained in {(end_train - start_train)/60:.2f} minutes.")

# --- 7. Prediction & Evaluation ---
print("Generating predictions and evaluating...")
predictions = {}
evaluation_results = {}
start_predict = time.time()

for i, target_col in enumerate(target_cols):
    original_col = original_cols[i] # Get the corresponding original name
    print(f"-- Evaluating {original_col} --")

    model = final_models[target_col]
    y_pred_series = model.predict(X_test)
    predictions[original_col] = y_pred_series # Store preds with original column name

    # Evaluate
    y_test_series = y_test[target_col]

    # Fix: Calculate RMSE manually
    mse = mean_squared_error(y_test_series, y_pred_series)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_series, y_pred_series)
    r2 = r2_score(y_test_series, y_pred_series)

    evaluation_results[original_col] = {'RMSE': rmse, 'MAE': mae, 'R2': r2}

    print(f'  Test RMSE: {rmse:.4f}')
    print(f'  Test MAE:  {mae:.4f}')
    print(f'  Test R2:   {r2:.4f}')

end_predict = time.time()
print(f"Prediction and evaluation finished in {(end_predict - start_predict):.2f} seconds.")

# --- 8. Output Generation ---
print("Generating final output file...")
pred_df = pd.DataFrame(predictions, index=X_test.index)

# Ensure columns are in the original order
pred_df = pred_df[original_cols]

# Save to CSV
pred_df.to_csv(OUTPUT_PREDICTIONS_FILE)
print(f"Predictions saved to {OUTPUT_PREDICTIONS_FILE}")

# Print summary of evaluation
print("\n--- Evaluation Summary ---")
eval_summary = pd.DataFrame(evaluation_results).T
print(eval_summary)
print("-------------------------")

print("\nScript finished.")
# %%