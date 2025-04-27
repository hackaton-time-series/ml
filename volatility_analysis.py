import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm
import sys

# --- Configuration ---
# <<< Make sure this points to your LATEST prediction file >>>
PREDICTIONS_FILE = 'predictions_abs_trig_Xvol_gapped_cv.csv'
RAW_DATA_FILE = 'data.csv'
OUTPUT_PLOT_FILE = 'volatility_vs_error_plot.png' # Output plot name

SERIES_TO_PLOT = ['Series1', 'Series2', 'Series3', 'Series4', 'Series5', 'Series6']
PLOT_PERIOD_LIMIT = '14D' # Limit time plot duration
# PLOT_PERIOD_LIMIT = None
IQR_MULTIPLIER = 1.5
VOLATILITY_WINDOW = 60 # Rolling window in minutes for volatility calculation

# --- Load Prediction Data ---
print(f"Loading predictions from: {PREDICTIONS_FILE}")
predictions = pd.read_csv(PREDICTIONS_FILE, parse_dates=['Date'], index_col='Date')

# --- Load and Clean Actual Data ---
print(f"Loading and cleaning actual data from: {RAW_DATA_FILE}")
actual_data = pd.read_csv(RAW_DATA_FILE, parse_dates=['Date'], index_col='Date')
original_cols = [col for col in actual_data.columns if col.startswith('Series')]
# Apply Cleaning
actual_data[original_cols] = actual_data[original_cols].interpolate(method='time')
for col in tqdm(original_cols, desc="Cleaning Actuals (IQR)", disable=True):
    Q1 = actual_data[col].quantile(0.25); Q3 = actual_data[col].quantile(0.75); IQR = Q3 - Q1
    lower_bound = Q1 - IQR_MULTIPLIER * IQR; upper_bound = Q3 + IQR_MULTIPLIER * IQR
    is_outlier = (actual_data[col] < lower_bound) | (actual_data[col] > upper_bound)
    actual_data[col] = actual_data[col].mask(is_outlier)
actual_data[original_cols] = actual_data[original_cols].interpolate(method='time')
actual_data.dropna(subset=original_cols, inplace=True)
inferred_freq = pd.infer_freq(actual_data.index); freq_to_use = 'min'
if inferred_freq:
    if 'MIN' in inferred_freq.upper() or inferred_freq.upper() == 'T': freq_to_use = 'min'
    else: freq_to_use = inferred_freq
actual_data = actual_data.asfreq(freq_to_use)
actual_data[original_cols] = actual_data[original_cols].interpolate(method='time')
actual_data.dropna(subset=original_cols, inplace=True)
print("Actual data cleaned.")
df_clean = actual_data # Use cleaned data

# --- Align Data & Calculate Errors & Volatility ---
common_cols = [col for col in SERIES_TO_PLOT if col in predictions.columns and col in df_clean.columns]
if not common_cols: raise ValueError("No common series columns found.")

pred_start_date = predictions.index.min(); pred_end_date = predictions.index.max()
print(f"Prediction time range: {pred_start_date} to {pred_end_date}")

actual_test_period = df_clean.loc[pred_start_date:pred_end_date, common_cols]
predictions_filtered = predictions.loc[pred_start_date:pred_end_date, common_cols]

if not actual_test_period.index.equals(predictions_filtered.index):
     print("Warning: Indices do not perfectly align. Using intersection.")
     common_index = predictions_filtered.index.intersection(actual_test_period.index)
     predictions_filtered = predictions_filtered.loc[common_index]
     actual_test_period = actual_test_period.loc[common_index]
     if common_index.empty: raise ValueError("Could not align indices.")
     print(f"Aligned range: {common_index.min()} to {common_index.max()}")
else:
     print("Indices successfully aligned.")

# Calculate Errors
errors = actual_test_period - predictions_filtered
abs_errors = errors.abs()

# Calculate Rolling Volatility on the *actual* test period data
print(f"Calculating {VOLATILITY_WINDOW}-min rolling volatility...")
rolling_std = actual_test_period.rolling(window=VOLATILITY_WINDOW, min_periods=VOLATILITY_WINDOW // 2).std()

# Apply plot period limit
plot_start_date = actual_test_period.index.min()
if PLOT_PERIOD_LIMIT:
    plot_end_date = plot_start_date + pd.Timedelta(PLOT_PERIOD_LIMIT)
    print(f"Limiting plot period to: {plot_start_date} to {plot_end_date}")
    abs_errors_plot = abs_errors.loc[plot_start_date:plot_end_date]
    rolling_std_plot = rolling_std.loc[plot_start_date:plot_end_date]
    if abs_errors_plot.empty or rolling_std_plot.empty:
         print(f"Warning: No data found in the specified plot limit '{PLOT_PERIOD_LIMIT}'.")
else:
    abs_errors_plot = abs_errors
    rolling_std_plot = rolling_std

# --- Create Plots ---
print("Generating volatility vs. error plots...")
valid_plot = not abs_errors_plot.empty and not rolling_std_plot.empty

if valid_plot:
    n_series = len(common_cols); n_cols = 2; n_rows = (n_series + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4), sharex=True)
    axes = axes.flatten(); plot_successful = False; last_plotted_index = -1

    for i, col in enumerate(common_cols):
        if col not in rolling_std_plot or col not in abs_errors_plot or \
           rolling_std_plot[col].isna().all() or abs_errors_plot[col].isna().all():
             print(f"Skipping plot for {col}, insufficient data.")
             if i < len(axes) and axes[i].figure == fig: fig.delaxes(axes[i])
             continue

        ax = axes[i]
        color1 = 'tab:blue'
        color2 = 'tab:red'

        # Plot Volatility on primary y-axis
        ax.plot(rolling_std_plot.index, rolling_std_plot[col], color=color1, label=f'Rolling Std Dev ({VOLATILITY_WINDOW}min)', linewidth=1.5)
        ax.set_ylabel('Rolling Std Dev', color=color1)
        ax.tick_params(axis='y', labelcolor=color1)
        ax.grid(True, linestyle='--', alpha=0.6)

        # Create secondary y-axis for Absolute Error
        ax2 = ax.twinx()
        ax2.plot(abs_errors_plot.index, abs_errors_plot[col], color=color2, label='Absolute Error |Act-Pred|', linewidth=1.0, alpha=0.6)
        ax2.set_ylabel('Absolute Error', color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)
        # Optional: Set ylim for error axis if needed, e.g., ax2.set_ylim(bottom=0)

        ax.set_title(f'{col} - Volatility vs. Absolute Error')
        # Combine legends from both axes
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper left')

        plot_successful = True
        last_plotted_index = i

    if plot_successful:
        print("Applying automatic date formatting...")
        fig.autofmt_xdate(rotation=30, ha='right')
    else:
         print("No data was plotted.")

    for j in range(last_plotted_index + 1, len(axes)):
         if j < len(axes) and axes[j].figure == fig: fig.delaxes(axes[j])

    fig.suptitle(f'Rolling {VOLATILITY_WINDOW}min Volatility vs. Absolute Prediction Error', fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])

    try: plt.savefig(OUTPUT_PLOT_FILE); print(f"Volatility plots saved to {OUTPUT_PLOT_FILE}")
    except Exception as e: print(f"Error saving plot: {e}")
    # plt.show()
else:
     print("Skipping plotting due to empty data after filtering/alignment.")

print("\nVolatility analysis script finished.")