# %%
# FINAL SCRIPT - Training on History, Predicting on Separate Test Set, Shifted Output
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error # Keep for Optuna objective
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# --- Configuration ---
TRAINING_DATA_FILE = 'Train_timeseries.csv' # Full historical data for training
TEST_DATA_FILE = 'Timeseries_six_test.csv' # Your specific test set file
# Final Output Names
OUTPUT_PREDICTIONS_FILE = 'predictions.csv' # Final prediction output
OUTPUT_IMPORTANCE_PLOT = 'feature_importances_final.png' # Feature importance plot

FORECAST_HORIZON = 240  # 4 hours = 240 minutes
N_SPLITS_CV = 5         # Number of folds for cross-validation
# <<< SET ADEQUATE TRIALS (e.g., 30+) FOR FINAL RUN! >>>
OPTUNA_TRIALS = 2
# TRAIN_SPLIT_RATIO is not used when predicting on a separate test file
IQR_MULTIPLIER = 1.5
CV_GAP = FORECAST_HORIZON

# --- 1. Load and Prepare TRAINING Data ---
print(f"Loading TRAINING data from {TRAINING_DATA_FILE}...")
try:
    train_data_raw = pd.read_csv(TRAINING_DATA_FILE, parse_dates=['Date'], index_col='Date')
    original_cols = [col for col in train_data_raw.columns if col.startswith('Series')]
    print(f"Raw Train data shape: {train_data_raw.shape}")
    if train_data_raw.empty: print("Error: Training data file empty."); sys.exit()
    if not isinstance(train_data_raw.index, pd.DatetimeIndex): print("Error: Index not DatetimeIndex."); sys.exit()
except Exception as e: print(f"Error loading training data: {e}"); sys.exit()

print("Cleaning and preprocessing TRAINING data...")
data_train = train_data_raw.copy()
# (Cleaning steps)
data_train[original_cols] = data_train[original_cols].interpolate(method='time')
for col in tqdm(original_cols, desc="Cleaning Train (IQR)", disable=True):
    Q1 = data_train[col].quantile(0.25); Q3 = data_train[col].quantile(0.75); IQR = Q3 - Q1
    lower_bound = Q1 - IQR_MULTIPLIER * IQR; upper_bound = Q3 + IQR_MULTIPLIER * IQR
    is_outlier = (data_train[col] < lower_bound) | (data_train[col] > upper_bound); data_train[col] = data_train[col].mask(is_outlier)
data_train[original_cols] = data_train[original_cols].interpolate(method='time')
data_train.dropna(subset=original_cols, inplace=True)
inferred_freq = pd.infer_freq(data_train.index); freq_to_use = 'min'
if inferred_freq:
    if 'MIN' in inferred_freq.upper() or inferred_freq.upper() == 'T': freq_to_use = 'min'
    else: freq_to_use = inferred_freq
print(f"Using frequency: {freq_to_use}")
data_train = data_train.asfreq(freq_to_use); data_train[original_cols] = data_train[original_cols].interpolate(method='time')
data_train.dropna(subset=original_cols, inplace=True); print(f"Cleaned Train data shape: {data_train.shape}")

print("Feature engineering on TRAINING data...")
df_train = data_train.copy()
# (Generate same features as before on df_train)
df_train['hour'] = df_train.index.hour; df_train['day_of_week'] = df_train.index.dayofweek; df_train['day_of_year'] = df_train.index.dayofyear
df_train['month'] = df_train.index.month; df_train['week_of_year'] = df_train.index.isocalendar().week.astype(int); df_train['year'] = df_train.index.year
df_train['is_weekend'] = (df_train.index.dayofweek >= 5).astype(int)
df_train['hour_sin'] = np.sin(2*np.pi*df_train['hour']/24.0); df_train['hour_cos'] = np.cos(2*np.pi*df_train['hour']/24.0)
df_train['dow_sin'] = np.sin(2*np.pi*df_train['day_of_week']/7.0); df_train['dow_cos'] = np.cos(2*np.pi*df_train['day_of_week']/7.0)
df_train['month_sin'] = np.sin(2*np.pi*df_train['month']/12.0); df_train['month_cos'] = np.cos(2*np.pi*df_train['month']/12.0)
new_feature_cols = ['hour', 'day_of_week', 'day_of_year', 'month', 'week_of_year', 'year', 'is_weekend',
                    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos']
lags_self = [1, 5, 15, 60, FORECAST_HORIZON, FORECAST_HORIZON + 15]; lags_cross = [1, 5]
for col in tqdm(original_cols, desc="Train Lag Feats", disable=True):
    for lag in lags_self: fname = f'{col}_lag_{lag}'; df_train[fname] = df_train[col].shift(lag); new_feature_cols.append(fname)
    for other_col in original_cols:
        if col == other_col: continue
        for lag in lags_cross: fname = f'{col}_X_{other_col}_lag_{lag}'; df_train[fname] = df_train[other_col].shift(lag); new_feature_cols.append(fname)
windows = [15, 60, FORECAST_HORIZON];
for col in tqdm(original_cols, desc="Train Rolling Feats", disable=True):
    for w in windows:
        roll = df_train[col].rolling(window=w, min_periods=3, closed='left')
        mfname = f'{col}_rollmean_{w}'; sfname = f'{col}_rollstd_{w}'; df_train[mfname] = roll.mean(); df_train[sfname] = roll.std()
        new_feature_cols.extend([mfname, sfname])
df_train = df_train.copy() # Defragment

print("Creating training targets...")
target_cols = []
for col in original_cols:
    target_col = f'{col}_target'; df_train[target_col] = df_train[col].shift(-FORECAST_HORIZON); target_cols.append(target_col)

# Drop NaNs from feature generation and target shift in training data
print(f"Train shape before NaNs drop: {df_train.shape}")
df_train.dropna(inplace=True)
print(f"Train shape after NaNs drop: {df_train.shape}")
if df_train.empty: raise ValueError("Training DataFrame empty after dropping NaNs.")

X_train = df_train[new_feature_cols]
y_train = df_train[target_cols]

# --- 2. Load and Prepare TEST Data ---
print(f"\nLoading TEST data from {TEST_DATA_FILE}...")
try:
    test_data_raw = pd.read_csv(TEST_DATA_FILE, parse_dates=['Date'], index_col='Date')
    test_index = test_data_raw.index # Keep original index for final output
    if not isinstance(test_index, pd.DatetimeIndex): raise ValueError("Test index not DatetimeIndex.")
    if test_index.empty: raise ValueError("Test index empty.")
    print(f"Test data shape: {test_data_raw.shape}")
except Exception as e: print(f"Error loading test data: {e}"); sys.exit()

print("Cleaning and preprocessing TEST data...")
data_test = test_data_raw.copy()
# IMPORTANT: Use the SAME cleaning logic as training data
data_test[original_cols] = data_test[original_cols].interpolate(method='time') # Only initial interpolation for test usually
# Typically, you would NOT remove outliers from the test set
# If you do, ensure it's done without lookahead bias (e.g., rolling IQR)
# For simplicity matching previous steps, we apply IQR based on test set stats (less ideal than robust method)
print(f"Applying IQR Outlier Masking to TEST data (use with caution)...")
for col in tqdm(original_cols, desc="Cleaning Test (IQR)", disable=True):
     Q1 = data_test[col].quantile(0.25); Q3 = data_test[col].quantile(0.75); IQR = Q3 - Q1
     lower_bound = Q1 - IQR_MULTIPLIER * IQR; upper_bound = Q3 + IQR_MULTIPLIER * IQR
     is_outlier = (data_test[col] < lower_bound) | (data_test[col] > upper_bound); data_test[col] = data_test[col].mask(is_outlier)
data_test[original_cols] = data_test[original_cols].interpolate(method='time')
# Do NOT drop NaNs here, we need to preserve the test set index

print("Ensuring test data frequency ('min')...")
data_test = data_test.reindex(pd.date_range(start=data_test.index.min(), end=data_test.index.max(), freq=freq_to_use), method=None) # Reindex first
data_test[original_cols] = data_test[original_cols].interpolate(method='time') # Interpolate after reindex
# Do not drop NaNs from test set if some remain - prediction loop should handle
print(f"Cleaned Test data shape: {data_test.shape}")

# --- 3. Feature Engineering for TEST Data ---
# Combine end of train data with test data for calculating features across boundary
max_lookback = max(lags_self + windows) # Max history needed for features
print(f"Max feature lookback needed: {max_lookback} minutes")
# Use cleaned training data ('data_train' from step 1)
train_history_for_features = data_train.iloc[-max_lookback:]

print(f"Combining train history ({len(train_history_for_features)} rows) with test data ({len(data_test)} rows) for feature calculation...")
df_combined_for_features = pd.concat([train_history_for_features[original_cols], data_test[original_cols]])

print("Starting feature engineering on combined data...")
df_test_features = df_combined_for_features.copy() # Base for features
# (Generate same features as for training, applied to this combined df)
df_test_features['hour'] = df_test_features.index.hour; df_test_features['day_of_week'] = df_test_features.index.dayofweek; df_test_features['day_of_year'] = df_test_features.index.dayofyear
df_test_features['month'] = df_test_features.index.month; df_test_features['week_of_year'] = df_test_features.index.isocalendar().week.astype(int); df_test_features['year'] = df_test_features.index.year
df_test_features['is_weekend'] = (df_test_features.index.dayofweek >= 5).astype(int)
df_test_features['hour_sin'] = np.sin(2*np.pi*df_test_features['hour']/24.0); df_test_features['hour_cos'] = np.cos(2*np.pi*df_test_features['hour']/24.0)
df_test_features['dow_sin'] = np.sin(2*np.pi*df_test_features['day_of_week']/7.0); df_test_features['dow_cos'] = np.cos(2*np.pi*df_test_features['day_of_week']/7.0)
df_test_features['month_sin'] = np.sin(2*np.pi*df_test_features['month']/12.0); df_test_features['month_cos'] = np.cos(2*np.pi*df_test_features['month']/12.0)
# Features calculated on combined data will have correct lookback across boundary
for col in tqdm(original_cols, desc="Test Lag Feats", disable=True):
    for lag in lags_self: fname = f'{col}_lag_{lag}'; df_test_features[fname] = df_test_features[col].shift(lag)
    for other_col in original_cols:
        if col == other_col: continue
        for lag in lags_cross: fname = f'{col}_X_{other_col}_lag_{lag}'; df_test_features[fname] = df_test_features[other_col].shift(lag)
for col in tqdm(original_cols, desc="Test Rolling Feats", disable=True):
    for w in windows:
        roll = df_test_features[col].rolling(window=w, min_periods=3, closed='left')
        mfname = f'{col}_rollmean_{w}'; sfname = f'{col}_rollstd_{w}'; df_test_features[mfname] = roll.mean(); df_test_features[sfname] = roll.std()
df_test_features = df_test_features.copy() # Defragment

# Select ONLY the rows corresponding to the original test_index for X_test
X_test = df_test_features.loc[test_index, new_feature_cols].copy()
print(f"Final X_test shape: {X_test.shape}")

# Handle NaNs created at the START of X_test due to feature lookback
print(f"NaNs count in initial X_test: {X_test.isna().sum().sum()}")
if X_test.isna().any().any():
    print("Forward/Backward filling NaNs in X_test features...")
    X_test.fillna(method='ffill', inplace=True)
    X_test.fillna(method='bfill', inplace=True)
    if X_test.isna().any().any(): print("Warning: NaNs still present after ffill/bfill. Filling with 0."); X_test.fillna(0, inplace=True)

# --- 4. Hyperparameter Optimization (on Training Data) ---
print(f"\nStarting hyperparameter optimization (Using CV gap={CV_GAP} on Training Data)...")
# (Optuna setup and objective function use X_train, y_train)
tscv = TimeSeriesSplit(n_splits=N_SPLITS_CV, gap=CV_GAP, test_size=FORECAST_HORIZON * 2)
tuning_target_col = 'Series5_target'; print(f"Tuning hyperparameters based on: {tuning_target_col}")
y_train_tuning = y_train[tuning_target_col]
# Objective function definition (as before)
def objective(trial):
    params = { 'objective': 'regression_l1', 'metric': 'rmse', 'boosting_type': 'gbdt','verbosity': -1, 'n_jobs': -1, 'seed': 42,
        'learning_rate': trial.suggest_float('learning_rate', 5e-4, 0.05, log=True), 'num_leaves': trial.suggest_int('num_leaves', 16, 256),
        'max_depth': trial.suggest_int('max_depth', 4, 10), 'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 50, 300),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 0.95), 'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 0.95),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7), 'lambda_l1': trial.suggest_float('lambda_l1', 1e-7, 15.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-7, 15.0, log=True), }
    cv_rmses = []; X_train_np=X_train.values; y_train_tuning_np=y_train_tuning.values
    for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_train_np)):
        if len(va_idx)==0: continue
        X_tr, X_va = X_train_np[tr_idx], X_train_np[va_idx]; y_tr, y_va = y_train_tuning_np[tr_idx], y_train_tuning_np[va_idx]
        if X_va.shape[0]==0 or y_va.shape[0]==0: continue
        dtrain=lgb.Dataset(X_tr, label=y_tr, feature_name=new_feature_cols); dvalid=lgb.Dataset(X_va, label=y_va, reference=dtrain, feature_name=new_feature_cols)
        gbm=lgb.train(params, dtrain, num_boost_round=1000, valid_sets=[dtrain, dvalid], callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        preds=gbm.predict(X_va); mse=mean_squared_error(y_va, preds); cv_rmses.append(np.sqrt(mse))
    if not cv_rmses: return float('inf')
    return np.mean(cv_rmses)
study = optuna.create_study(direction='minimize')
start_optuna = time.time()
study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True)
end_optuna = time.time(); print(f"Optuna finished in {(end_optuna - start_optuna)/60:.2f} minutes.")
print("🎯 Best Params:", study.best_params); print(f"🎯 Best CV RMSE ({tuning_target_col} with gap={CV_GAP}): {study.best_value:.4f}")

# --- 5. Final Model Training (on Training Data) ---
print("Training final models on ALL training data...")
best_params = study.best_params
best_params.update({'objective': 'regression_l1', 'metric': 'rmse', 'verbosity': -1, 'n_jobs': -1, 'seed': 42})
final_models = {}
start_train = time.time()
# Determine optimal rounds using early stopping on a validation split of TRAINING data
print(f"Finding optimal boost rounds using {tuning_target_col}...")
val_size = int(len(X_train) * 0.1);
if val_size >= len(X_train) or val_size==0: val_size = max(1, min(len(X_train)-1, int(len(X_train) * 0.05)))
X_train_part, X_val_part = X_train[:-val_size], X_train[-val_size:]
y_train_tuning_part, y_val_tuning_part = y_train_tuning[:-val_size], y_train_tuning[-val_size:]
optimal_boost_rounds = 100
if X_val_part.empty or y_val_tuning_part.empty: print("Warning: Validation set empty. Using fixed 100 rounds.")
else:
    dtrain_opt = lgb.Dataset(X_train_part, label=y_train_tuning_part, feature_name=new_feature_cols)
    dvalid_opt = lgb.Dataset(X_val_part, label=y_val_tuning_part, reference=dtrain_opt, feature_name=new_feature_cols)
    temp_gbm = lgb.train(best_params, dtrain_opt, num_boost_round=3000, valid_sets=[dtrain_opt, dvalid_opt], callbacks=[lgb.early_stopping(150, verbose=True), lgb.log_evaluation(100)])
    optimal_boost_rounds = temp_gbm.best_iteration; print(f"Optimal boosting rounds determined: {optimal_boost_rounds}")
    if optimal_boost_rounds <= 0: optimal_boost_rounds = 100
# Train final models on FULL training data
for target_col in tqdm(target_cols, desc="Training Final Models"):
    series_y_train = y_train[target_col]
    dtrain_full = lgb.Dataset(X_train, label=series_y_train, feature_name=new_feature_cols)
    final_gbm = lgb.train(best_params, dtrain_full, num_boost_round=optimal_boost_rounds)
    final_models[target_col] = final_gbm # Keyed by 'SeriesX_target'
end_train = time.time(); print(f"Final models trained in {(end_train - start_train)/60:.2f} minutes.")

# --- 6. Plot Feature Importances ---
print("Plotting feature importances...")
# (Plotting code as before) ...
n_series = len(original_cols); fig, axes = plt.subplots(n_series, 1, figsize=(14, n_series * 7))
if n_series == 1: axes = [axes]
for i, target_col in enumerate(target_cols):
    original_col = original_cols[i]; model = final_models[target_col]
    importance_df = pd.DataFrame({'feature': model.feature_name(), 'importance': model.feature_importance(importance_type='gain')}).sort_values('importance', ascending=False).head(30)
    sns.barplot(x='importance', y='feature', data=importance_df, ax=axes[i], palette='viridis', hue='feature', legend=False)
    axes[i].set_title(f'Feature Importance (Top 30 Gain) for {original_col}'); axes[i].set_xlabel('Importance (Gain)'); axes[i].set_ylabel('Feature')
plt.tight_layout(); plt.savefig(OUTPUT_IMPORTANCE_PLOT); print(f"Feature importance plot saved to {OUTPUT_IMPORTANCE_PLOT}")

# --- 7. Prediction Generation & Shifting ---
print("Generating final predictions for the test set and shifting...")
predictions_shifted_dict = {}
start_predict = time.time()
for i, target_col in enumerate(target_cols):
    original_col = original_cols[i]; print(f"-- Predicting for {original_col} --")
    model = final_models[target_col]
    # Predict on X_test (features aligned with test_index)
    # y_pred has index t (from test_index), value is pred[t+H]
    y_pred = pd.Series(model.predict(X_test), index=X_test.index)
    # Shift forward to align with target time
    y_pred_shifted = y_pred.shift(FORECAST_HORIZON)
    predictions_shifted_dict[original_col] = y_pred_shifted
end_predict = time.time(); print(f"\nPrediction generation finished in {(end_predict - start_predict):.2f} seconds.")

# --- 8. Output Generation (Matching Test Set Index) ---
print(f"Generating final output file '{OUTPUT_PREDICTIONS_FILE}'...")
pred_df_intermediate = pd.DataFrame(predictions_shifted_dict)
# Reindex using the original test_index loaded from the file
final_pred_df = pred_df_intermediate.reindex(test_index, fill_value=np.nan)
final_pred_df = final_pred_df[original_cols] # Ensure original column order
print(f"Output predictions shape: {final_pred_df.shape}")
if not final_pred_df.index.equals(test_index): print(f"Warning: Output index does not match input test index!")
else: print("Output index successfully matches input test index.")
print(f"NaNs count in first {FORECAST_HORIZON+1} rows of Series1 predictions: {final_pred_df['Series1'].iloc[:FORECAST_HORIZON+1].isna().sum()}")
final_pred_df.to_csv(OUTPUT_PREDICTIONS_FILE); print(f"Predictions saved to {OUTPUT_PREDICTIONS_FILE}")

# --- REMOVED Evaluation Section ---
print("\nScript finished.")
# %%