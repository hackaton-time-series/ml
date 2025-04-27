import pandas as pd
import numpy as np # Import numpy
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm

# --- Configuration ---
PREDICTIONS_FILE = 'predictions.csv' # <--- MAKE SURE this is your prediction file name
RAW_DATA_FILE = 'data.csv'
OUTPUT_PLOT_FILE = 'prediction_vs_actual_plots_v2.png' # New plot name

SERIES_TO_PLOT = ['Series1', 'Series2', 'Series3', 'Series4', 'Series5', 'Series6']
PLOT_PERIOD_LIMIT = '7D'
# PLOT_PERIOD_LIMIT = None
IQR_MULTIPLIER = 1.5

# --- Load Prediction Data ---
print(f"Loading predictions from: {PREDICTIONS_FILE}")
predictions = pd.read_csv(PREDICTIONS_FILE, parse_dates=['Date'], index_col='Date')

# --- Load and Clean Actual Data ---
print(f"Loading and cleaning actual data from: {RAW_DATA_FILE}")
actual_data = pd.read_csv(RAW_DATA_FILE, parse_dates=['Date'], index_col='Date')
original_cols = [col for col in actual_data.columns if col.startswith('Series')]
print("Cleaning actual data (using same steps as training)...")
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

# --- Align Data ---
common_cols = [col for col in SERIES_TO_PLOT if col in predictions.columns and col in actual_data.columns]
if not common_cols: raise ValueError("No common series columns found.")

pred_start_date = predictions.index.min(); pred_end_date = predictions.index.max()
print(f"Prediction time range: {pred_start_date} to {pred_end_date}")

actual_test_period = actual_data.loc[pred_start_date:pred_end_date, common_cols]
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

# Apply plot period limit
plot_start_date = actual_test_period.index.min()
if PLOT_PERIOD_LIMIT:
    plot_end_date = plot_start_date + pd.Timedelta(PLOT_PERIOD_LIMIT)
    print(f"Limiting plot period to: {plot_start_date} to {plot_end_date}")
    actual_plot_data = actual_test_period.loc[plot_start_date:plot_end_date]
    predictions_plot_data = predictions_filtered.loc[plot_start_date:plot_end_date]
    if actual_plot_data.empty or predictions_plot_data.empty:
         print(f"Warning: No data in plot limit '{PLOT_PERIOD_LIMIT}'.")
else:
    actual_plot_data = actual_test_period
    predictions_plot_data = predictions_filtered

# --- Add Index Checks ---
print("\nIndex check before plotting:")
valid_plot = True
if actual_plot_data.empty:
    print("Error: Actual plot data is empty after filtering."); valid_plot = False
else:
    print(f"Actual index type: {type(actual_plot_data.index)}, NaNs: {actual_plot_data.index.isna().sum()}, Min: {actual_plot_data.index.min()}, Max: {actual_plot_data.index.max()}")
if predictions_plot_data.empty:
     print("Error: Predictions plot data is empty after filtering."); valid_plot = False
else:
     print(f"Pred index type: {type(predictions_plot_data.index)}, NaNs: {predictions_plot_data.index.isna().sum()}, Min: {predictions_plot_data.index.min()}, Max: {predictions_plot_data.index.max()}")
print("-" * 20)
# --- End Index Checks ---


# --- Create Plots ---
if valid_plot:
    print("Generating plots...")
    n_series = len(common_cols); n_cols = 2; n_rows = (n_series + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4), sharex=True)
    axes = axes.flatten()
    plot_successful = False
    last_plotted_index = -1

    for i, col in enumerate(common_cols):
        if col not in actual_plot_data or col not in predictions_plot_data or actual_plot_data[col].empty or predictions_plot_data[col].empty:
             print(f"Skipping plot for {col}, insufficient data in period.")
             if i < len(axes) and axes[i].figure == fig: fig.delaxes(axes[i])
             continue

        ax = axes[i]
        actual_vals = actual_plot_data[col]
        predicted_vals = predictions_plot_data[col]

        actual_vals.plot(ax=ax, label='Actual', color='blue', linewidth=1.0, alpha=0.8)
        predicted_vals.plot(ax=ax, label='Predicted', color='red', linewidth=1.0, alpha=0.8)

        ax.set_title(f'{col} - Actual vs. Predicted')
        ax.set_ylabel('Value'); ax.legend(); ax.grid(True, linestyle='--', alpha=0.6)
        plot_successful = True
        last_plotted_index = i # Track the last successfully plotted axis index

    # --- Use Matplotlib's automatic date formatting ---
    if plot_successful:
        print("Applying automatic date formatting...")
        # Remove previous specific formatters/locators if they caused issues
        # ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))
        # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        # plt.xticks(rotation=30) # Replaced by autofmt_xdate

        # Let Matplotlib handle the formatting and rotation automatically
        fig.autofmt_xdate(rotation=30, ha='right')
    else:
         print("No data was plotted.")
    # --- End date formatting change ---

    # Hide any unused subplots
    for j in range(last_plotted_index + 1, len(axes)):
         if j < len(axes) and axes[j].figure == fig: fig.delaxes(axes[j])

    fig.suptitle('Time Series Predictions vs. Actual Values (Test Period)', fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])

    try:
        plt.savefig(OUTPUT_PLOT_FILE)
        print(f"Plots saved to {OUTPUT_PLOT_FILE}")
    except Exception as e: print(f"Error saving plot: {e}")
    # plt.show()
    print("Visualization script finished.")
else:
     print("Skipping plotting due to empty or invalid data.")