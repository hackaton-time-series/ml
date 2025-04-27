# %%
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Ensure plots are displayed inline if using Jupyter/IPython, otherwise saved to file
# %matplotlib inline

# --- Configuration ---
RAW_DATA_FILE = 'data.csv'
# Final Output Names
OUTPUT_PREDICTIONS_FILE = 'predictions_lgbm_final.csv'
OUTPUT_IMPORTANCE_PLOT = 'feature_importances_lgbm_final.png'

FORECAST_HORIZON = 240  # 4 hours = 240 minutes
N_SPLITS_CV = 5         # Number of folds for cross-validation
# <<< CRITICAL: Use adequate trials (e.g., 30+) for final run! >>>
OPTUNA_TRIALS = 30      # Recommended setting for final run
TRAIN_SPLIT_RATIO = 0.9 # 90% train, 10% test
IQR_MULTIPLIER = 1.5    # Multiplier for IQR outlier detection
CV_GAP = FORECAST_HORIZON # Gap between train/validation in CV folds

# --- 1. Load Raw Data ---
print(f"Loading raw data from {RAW_DATA_FILE}...")
try:
    data = pd.read_csv(RAW_DATA_FILE, parse_dates=['Date'], index_col='Date')
    original_cols = [col for col in data.columns if col.startswith('Series')]
    print(f"Raw data shape: {data.shape}")
    if data.empty: print("Error: Data file empty."); sys.exit()
    if not isinstance(data.index, pd.DatetimeIndex): print("Error: Index not DatetimeIndex."); sys.exit()
except Exception as e: print(f"Error loading data: {e}"); sys.exit()

# --- 2. Cleaning & Preprocessing ---
print("Starting cleaning and preprocessing...")
print("Step 2a: Initial time interpolation...")
data[original_cols] = data[original_cols].interpolate(method='time')
print(f"Step 2b: IQR Outlier Masking (multiplier={IQR_MULTIPLIER})...")
for col in tqdm(original_cols, desc="Detecting Outliers", disable=True):
    Q1 = data[col].quantile(0.25); Q3 = data[col].quantile(0.75); IQR = Q3 - Q1
    lower_bound = Q1 - IQR_MULTIPLIER * IQR; upper_bound = Q3 + IQR_MULTIPLIER * IQR
    is_outlier = (data[col] < lower_bound) | (data[col] > upper_bound)
    data[col] = data[col].mask(is_outlier)
print("Step 2c: Time interpolation for outliers...")
data[original_cols] = data[original_cols].interpolate(method='time')
data.dropna(subset=original_cols, inplace=True)
print("Step 2d: Ensuring data frequency ('min')...")
inferred_freq = pd.infer_freq(data.index); freq_to_use = 'min'
if inferred_freq:
    if 'MIN' in inferred_freq.upper() or inferred_freq.upper() == 'T': freq_to_use = 'min'
    else: freq_to_use = inferred_freq
print(f"Using frequency: {freq_to_use}")
data = data.asfreq(freq_to_use)
data[original_cols] = data[original_cols].interpolate(method='time') # Interpolate again after asfreq
data.dropna(subset=original_cols, inplace=True) # Final check
print(f"Cleaned data shape: {data.shape}")
print("--- End Cleaning ---")

# --- 3. Feature Engineering (Simplified + Trig Features) ---
print("Starting feature engineering (Simplified + Trig Features)...")
df = data.copy()

# Calendar features (Basic + Trig)
df['hour'] = df.index.hour; df['day_of_week'] = df.index.dayofweek; df['day_of_year'] = df.index.dayofyear
df['month'] = df.index.month; df['week_of_year'] = df.index.isocalendar().week.astype(int); df['year'] = df.index.year
df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
df['hour_sin'] = np.sin(2*np.pi*df['hour']/24.0); df['hour_cos'] = np.cos(2*np.pi*df['hour']/24.0)
df['dow_sin'] = np.sin(2*np.pi*df['day_of_week']/7.0); df['dow_cos'] = np.cos(2*np.pi*df['day_of_week']/7.0)
df['month_sin'] = np.sin(2*np.pi*df['month']/12.0); df['month_cos'] = np.cos(2*np.pi*df['month']/12.0)
new_feature_cols = ['hour', 'day_of_week', 'day_of_year', 'month', 'week_of_year', 'year', 'is_weekend',
                    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos']

# Lag features (Self and Simple Cross)
lags_self = [1, 5, 15, 60, FORECAST_HORIZON, FORECAST_HORIZON + 15]
lags_cross = [1, 5] # Keep only simple, short cross-lags
print("Generating Lag Features...")
for col in tqdm(original_cols, disable=True):
    # Ensure lag 240 exists for potential baseline comparison (though baseline calc is removed)
    if FORECAST_HORIZON not in lags_self: lags_self.append(FORECAST_HORIZON)
    for lag in lags_self:
        fname = f'{col}_lag_{lag}'; df[fname] = df[col].shift(lag); new_feature_cols.append(fname)
    # Include simple cross-lags
    for other_col in original_cols:
        if col == other_col: continue
        for lag in lags_cross:
             fname = f'{col}_X_{other_col}_lag_{lag}'; df[fname] = df[other_col].shift(lag); new_feature_cols.append(fname)

# Rolling window features (Self only)
windows = [15, 60, FORECAST_HORIZON]
print("Generating Self-Rolling Features...")
for col in tqdm(original_cols, disable=True):
    for w in windows:
        roll = df[col].rolling(window=w, min_periods=3, closed='left')
        mfname = f'{col}_rollmean_{w}'; sfname = f'{col}_rollstd_{w}'
        df[mfname] = roll.mean(); df[sfname] = roll.std()
        new_feature_cols.extend([mfname, sfname])

print(f"Total features generated: {len(new_feature_cols)}") # Should be ~145
print("De-fragmenting DataFrame...")
df = df.copy()

# --- 4. Target Creation (ABSOLUTE TARGET) ---
print("Creating ABSOLUTE target variables...")
target_cols = []
for col in original_cols:
    target_col = f'{col}_target'
    df[target_col] = df[col].shift(-FORECAST_HORIZON)
    target_cols.append(target_col)

# --- 5. Handle NaNs & Split Data ---
print(f"Shape before dropping NaNs from features/targets: {df.shape}")
df.dropna(inplace=True)
print(f"Shape after dropping NaNs: {df.shape}")
if df.empty: raise ValueError("DataFrame empty after dropping NaNs.")
n = int(len(df) * TRAIN_SPLIT_RATIO)
train_df, test_df = df.iloc[:n], df.iloc[n:]
if test_df.empty: raise ValueError("Test set empty after split.")
X_train = train_df[new_feature_cols]
X_test = test_df[new_feature_cols]
y_train = train_df[target_cols]
y_test = test_df[target_cols]
print(f"Train shape: X={X_train.shape}, y={y_train.shape}")
print(f"Test shape: X={X_test.shape}, y={y_test.shape}")

# --- 6. Hyperparameter Optimization (with Gapped CV) ---
print(f"Starting hyperparameter optimization (Using CV gap={CV_GAP})...")
tscv = TimeSeriesSplit(n_splits=N_SPLITS_CV, gap=CV_GAP, test_size=FORECAST_HORIZON * 2)
tuning_target_col = 'Series5_target'
print(f"Tuning hyperparameters based on: {tuning_target_col}")
y_train_tuning = y_train[tuning_target_col]

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
study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True) # Use 30+ trials
end_optuna = time.time()
print(f"Optuna finished in {(end_optuna - start_optuna)/60:.2f} minutes.")
print("🎯 Best Params:", study.best_params)
print(f"🎯 Best CV RMSE ({tuning_target_col} with gap={CV_GAP}): {study.best_value:.4f}")

# --- 7. Final Model Training ---
print("Training final models for each series...")
best_params = study.best_params
best_params.update({'objective': 'regression_l1', 'metric': 'rmse', 'verbosity': -1, 'n_jobs': -1, 'seed': 42})
final_models = {}
start_train = time.time()
print(f"Finding optimal boost rounds using {tuning_target_col}...")
val_size = int(len(X_train) * 0.1)
if val_size >= len(X_train): val_size = max(1, int(len(X_train) * 0.05))
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
for target_col in tqdm(target_cols, desc="Training Final Models"):
    series_y_train = y_train[target_col]
    dtrain_full = lgb.Dataset(X_train, label=series_y_train, feature_name=new_feature_cols)
    final_gbm = lgb.train(best_params, dtrain_full, num_boost_round=optimal_boost_rounds)
    final_models[target_col] = final_gbm
end_train = time.time()
print(f"Final models trained in {(end_train - start_train)/60:.2f} minutes.")

# --- 8. Plot Feature Importances ---
print("Plotting feature importances...")
n_series = len(original_cols); fig, axes = plt.subplots(n_series, 1, figsize=(14, n_series * 7))
if n_series == 1: axes = [axes]
for i, target_col in enumerate(target_cols):
    original_col = original_cols[i]; model = final_models[target_col]
    importance_df = pd.DataFrame({'feature': model.feature_name(), 'importance': model.feature_importance(importance_type='gain')}).sort_values('importance', ascending=False).head(30)
    sns.barplot(x='importance', y='feature', data=importance_df, ax=axes[i], palette='viridis', hue='feature', legend=False)
    axes[i].set_title(f'Feature Importance (Top 30 Gain) for {original_col}'); axes[i].set_xlabel('Importance (Gain)'); axes[i].set_ylabel('Feature')
plt.tight_layout(); plt.savefig(OUTPUT_IMPORTANCE_PLOT); print(f"Feature importance plot saved to {OUTPUT_IMPORTANCE_PLOT}")

# --- 9. Prediction & Evaluation (LGBM Only) ---
print("Generating predictions and evaluating...")
predictions_final = {}
evaluation_results = {}
start_predict = time.time()
def symmetric_mean_absolute_percentage_error(y_true, y_pred, epsilon=1e-10):
     y_true, y_pred = np.array(y_true), np.array(y_pred)
     numerator = np.abs(y_pred - y_true); denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
     denominator = np.maximum(denominator, epsilon); return np.mean(numerator / denominator) * 100

for i, target_col in enumerate(target_cols):
    original_col = original_cols[i]
    print(f"\n-- Evaluating {original_col} --")
    model = final_models[target_col]; y_pred_lgbm = model.predict(X_test)
    predictions_final[original_col] = y_pred_lgbm; y_test_abs = y_test[target_col]
    print("  LightGBM Model:")
    mse = mean_squared_error(y_test_abs, y_pred_lgbm); rmse = np.sqrt(mse); mae = mean_absolute_error(y_test_abs, y_pred_lgbm)
    r2 = r2_score(y_test_abs, y_pred_lgbm); smape = symmetric_mean_absolute_percentage_error(y_test_abs, y_pred_lgbm)
    mean_abs_actual = np.mean(np.abs(y_test_abs)); mae_ratio = (mae / mean_abs_actual) * 100 if mean_abs_actual > 1e-10 else np.inf
    evaluation_results[original_col] = {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'sMAPE (%)': smape, 'MAE/Mean (%)': mae_ratio}
    print(f'    RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}, sMAPE: {smape:.2f}%, MAE/Mean: {mae_ratio:.2f}%')

    # --- REMOVED Naive Baseline Calculation ---

end_predict = time.time(); print(f"\nPrediction and evaluation finished in {(end_predict - start_predict):.2f} seconds.")

# --- 10. Output Generation ---
print("Generating final output file...")
pred_df = pd.DataFrame(predictions_final, index=X_test.index); pred_df = pred_df[original_cols]
pred_df.to_csv(OUTPUT_PREDICTIONS_FILE); print(f"Predictions saved to {OUTPUT_PREDICTIONS_FILE}")
print("\n--- LGBM Evaluation Summary ---"); eval_summary = pd.DataFrame(evaluation_results).T
pd.options.display.float_format = '{:.4f}'.format; print(eval_summary); print("------------------------------")
# --- REMOVED Naive Baseline Summary Printout ---
print("\nScript finished.")
# %%