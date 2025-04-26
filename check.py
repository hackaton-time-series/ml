# training.py

import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

# 1️⃣ Load cleaned data
data = pd.read_csv('data_clean.csv', parse_dates=['Date'], index_col='Date')

# 2️⃣ Calendar features
data['hour'] = data.index.hour
data['day_of_week'] = data.index.dayofweek
data['month'] = data.index.month
data['is_weekend'] = (data.index.dayofweek >= 5).astype(int)

# 3️⃣ Lagged & rolling stats
lags = [1, 5, 15, 60]
windows = [5, 15, 60]

lagged = {f'{col}_lag_{lag}': data[col].shift(lag) for lag in lags for col in data.columns[:6]}

roll = {}
for w in windows:
    for col in data.columns[:6]:
        roll[f'{col}_rollmean_{w}'] = data[col].rolling(window=w, min_periods=1).mean()
        roll[f'{col}_rollstd_{w}'] = data[col].rolling(window=w, min_periods=1).std()

df = pd.concat([data, pd.DataFrame(lagged), pd.DataFrame(roll)], axis=1).dropna()

# 4️⃣ Create 4-hour-ahead targets
h = 240
for col in data.columns[:6]:
    df[f'{col}_target'] = df[col].shift(-h)
df = df.dropna()

# 5️⃣ Time-based split
n = int(len(df) * 0.8)
X = df.drop([c for c in df.columns if c.endswith('_target')], axis=1)
y = df['Series5_target']

X_train, X_test = X.iloc[:n], X.iloc[n:]
y_train, y_test = y.iloc[:n], y.iloc[n:]

# 6️⃣ Define time-series CV
tscv = TimeSeriesSplit(n_splits=5, gap=0, test_size=h)

def objective(trial):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 16, 256),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 200),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
    }
    cv_rmses = []
    for tr_idx, va_idx in tscv.split(X_train):
        dtrain = lgb.Dataset(X_train.iloc[tr_idx], label=y_train.iloc[tr_idx])
        dvalid = lgb.Dataset(X_train.iloc[va_idx], label=y_train.iloc[va_idx])
        gbm = lgb.train(
            params,
            dtrain,
            num_boost_round=500,
            valid_sets=[dvalid],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(0)
            ]
        )
        preds = gbm.predict(X_train.iloc[va_idx])
        cv_rmses.append(mean_squared_error(y_train.iloc[va_idx], preds))
    return np.mean(cv_rmses)

# 7️⃣ Run Optuna study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30)

print("🎯 Best Params:", study.best_params)
print("🎯 Best CV RMSE:", study.best_value)

# 8️⃣ Retrain final model
best_params = study.best_params
best_params.update({'objective': 'regression', 'metric': 'rmse', 'verbosity': -1})

dtrain_full = lgb.Dataset(X_train, label=y_train)
final_gbm = lgb.train(
    best_params,
    dtrain_full,
    num_boost_round=500
)

# 9️⃣ Predict & evaluate
y_pred = final_gbm.predict(X_test)
rmse = mean_squared_error(y_test, y_pred)
print(f'✅ Final Test RMSE for Series5: {rmse:.4f}')

# 🔟 Save predictions
pd.Series(y_pred, index=y_test.index, name='pred').to_csv('series5_preds_final.csv')
