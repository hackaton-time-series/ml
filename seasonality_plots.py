import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # Use seaborn for potentially nicer plots
from tqdm import tqdm
import sys

# Configuration
RAW_DATA_FILE = 'data.csv'
IQR_MULTIPLIER = 1.5
OUTPUT_PLOT_FILE = 'seasonality_plots.png'

print(f"Python version: {sys.version}")
print(f"Pandas version: {pd.__version__}")
print(f"Matplotlib version: {plt.matplotlib.__version__}")
print(f"Seaborn version: {sns.__version__}")

# --- 1. Load and Clean Data (Same as before) ---
print(f"Loading raw data from {RAW_DATA_FILE}...")
try:
    data = pd.read_csv(RAW_DATA_FILE, parse_dates=['Date'], index_col='Date')
    original_cols = [col for col in data.columns if col.startswith('Series')]
    print(f"Raw data shape: {data.shape}")
    if data.empty: print("Error: Data file empty."); exit()
    if not isinstance(data.index, pd.DatetimeIndex): print("Error: Index not DatetimeIndex."); exit()
except Exception as e: print(f"Error loading data: {e}"); exit()

print("Cleaning data (IQR method)...")
data[original_cols] = data[original_cols].interpolate(method='time')
for col in tqdm(original_cols, desc="Cleaning (IQR)", disable=True):
    Q1 = data[col].quantile(0.25); Q3 = data[col].quantile(0.75); IQR = Q3 - Q1
    lower_bound = Q1 - IQR_MULTIPLIER * IQR; upper_bound = Q3 + IQR_MULTIPLIER * IQR
    is_outlier = (data[col] < lower_bound) | (data[col] > upper_bound)
    data[col] = data[col].mask(is_outlier)
data[original_cols] = data[original_cols].interpolate(method='time')
data.dropna(subset=original_cols, inplace=True)
inferred_freq = pd.infer_freq(data.index); freq_to_use = 'min'
if inferred_freq:
    if 'MIN' in inferred_freq.upper() or inferred_freq.upper() == 'T': freq_to_use = 'min'
    else: freq_to_use = inferred_freq
data = data.asfreq(freq_to_use)
data[original_cols] = data[original_cols].interpolate(method='time')
data.dropna(subset=original_cols, inplace=True)
print(f"Cleaned data shape: {data.shape}")
if data.empty: print("Error: Data empty after cleaning."); exit()
df_clean = data # Use this cleaned data

# --- 2. Calculate Seasonality Aggregations ---
print("\nCalculating seasonality...")
# Add time components
df_clean['hour'] = df_clean.index.hour
df_clean['dayofweek'] = df_clean.index.dayofweek # Monday=0, Sunday=6
df_clean['weekday_name'] = df_clean.index.day_name()

# Group by hour
hourly_avg = df_clean.groupby('hour')[original_cols].mean()

# Group by day of week
daily_avg = df_clean.groupby('dayofweek')[original_cols].mean()
# Reindex to ensure correct day order for plotting if needed
daily_avg = daily_avg.reindex(range(7)) # Ensure 0-6 index
day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"] # For labels

# --- 3. Plot Seasonality ---
print("Generating seasonality plots...")
n_series = len(original_cols)
fig, axes = plt.subplots(n_series, 2, figsize=(14, n_series * 3.5)) # Two plots per series

for i, col in enumerate(original_cols):
    # Hourly Plot
    ax_hour = axes[i, 0]
    hourly_avg[col].plot(ax=ax_hour, marker='.')
    ax_hour.set_title(f'{col} - Average by Hour of Day')
    ax_hour.set_xlabel('Hour (0-23)')
    ax_hour.set_ylabel('Average Value')
    ax_hour.grid(True, linestyle='--', alpha=0.6)
    ax_hour.set_xticks(range(0, 24, 2)) # Show ticks every 2 hours

    # Daily Plot
    ax_day = axes[i, 1]
    daily_avg[col].plot(ax=ax_day, marker='.')
    ax_day.set_title(f'{col} - Average by Day of Week')
    ax_day.set_xlabel('Day of Week')
    ax_day.set_ylabel('Average Value')
    ax_day.grid(True, linestyle='--', alpha=0.6)
    ax_day.set_xticks(range(7))
    ax_day.set_xticklabels(day_names)

plt.tight_layout()
plt.savefig(OUTPUT_PLOT_FILE)
print(f"Seasonality plots saved to {OUTPUT_PLOT_FILE}")
# plt.show()

print("\nSeasonality analysis finished.")