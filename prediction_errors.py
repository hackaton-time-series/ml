import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm
import sys

# --- Configuration ---
# <<< Make sure this points to your LATEST prediction file >>>
PREDICTIONS_FILE = 'predictions_abs_trig_features_gapped_cv.csv'
RAW_DATA_FILE = 'data.csv' # Use raw data for actuals
OUTPUT_PLOT_FILE = 'prediction_errors_plot.png' # Output plot name

SERIES_TO_PLOT = ['Series1', 'Series2', 'Series3', 'Series4', 'Series5', 'Series6']
PLOT_PERIOD_LIMIT = '14D' # Example: Plot errors for first 14 days of test set
# PLOT_PERIOD_LIMIT = None # Uncomment to plot all
IQR_MULTIPLIER = 1.5

# --- Load Prediction Data ---
print(f"Loading predictions from: {PREDICTIONS_FILE}")
predictions = pd.read_csv(PREDICTIONS_FILE, parse_dates=['Date'], index_col='Date')

# --- Load and Clean Actual Data (Replicate steps from training script) ---
print(f"Loading and cleaning actual data from: {RAW_DATA_FILE}")
actual_data = pd.read_csv(RAW_DATA_FILE, parse_dates=['Date'], index_col='Date')
original_cols = [col for col in actual_data.columns if col.startswith('Series')]
# Apply Cleaning (Interpolate -> IQR -> Interpolate -> Freq)
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

# --- Align Data & Calculate Errors ---
common_cols = [col for col in SERIES_TO_PLOT if col in predictions.columns and col in actual_data.columns]
if not common_cols: raise ValueError("No common series columns found.")

pred_start_date = predictions.index.min(); pred_end_date = predictions.index.max()
print(f"Prediction time range: {pred_start_date} to {pred_end_date}")

actual_test_period = actual_data.loc[pred_start_date:pred_end_date, common_cols]
predictions_filtered = predictions.loc[pred_start_date:pred_end_date, common_cols]

# Align indices if necessary
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

# Apply optional time limit for plotting
plot_start_date = errors.index.min()
if PLOT_PERIOD_LIMIT:
    plot_end_date = plot_start_date + pd.Timedelta(PLOT_PERIOD_LIMIT)
    print(f"Limiting plot period to: {plot_start_date} to {plot_end_date}")
    errors_plot_data = errors.loc[plot_start_date:plot_end_date]
    if errors_plot_data.empty :
         print(f"Warning: No error data found in the specified plot limit '{PLOT_PERIOD_LIMIT}'.")
else:
    errors_plot_data = errors

# --- Create Plots ---
print("Generating error plots...")
valid_plot = not errors_plot_data.empty

if valid_plot:
    n_series = len(common_cols); n_cols = 2; n_rows = (n_series + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 3.5), sharex=True)
    axes = axes.flatten()
    plot_successful = False; last_plotted_index = -1

    for i, col in enumerate(common_cols):
        if col not in errors_plot_data or errors_plot_data[col].empty:
             print(f"Skipping plot for {col}, insufficient data.")
             if i < len(axes) and axes[i].figure == fig: fig.delaxes(axes[i])
             continue

        ax = axes[i]
        errors_plot_data[col].plot(ax=ax, label='Error (Actual - Predicted)', color='green', linewidth=1.0)
        ax.axhline(0, color='red', linestyle='--', linewidth=0.8, label='Zero Error') # Add horizontal line at 0

        ax.set_title(f'{col} - Prediction Error Over Time')
        ax.set_ylabel('Error')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        plot_successful = True
        last_plotted_index = i

    if plot_successful:
        print("Applying automatic date formatting...")
        fig.autofmt_xdate(rotation=30, ha='right')
    else:
         print("No data was plotted.")

    for j in range(last_plotted_index + 1, len(axes)):
         if j < len(axes) and axes[j].figure == fig: fig.delaxes(axes[j])

    fig.suptitle('Prediction Errors (Actual - Predicted) Over Time', fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])

    try: plt.savefig(OUTPUT_PLOT_FILE); print(f"Error plots saved to {OUTPUT_PLOT_FILE}")
    except Exception as e: print(f"Error saving plot: {e}")
    # plt.show()
else:
     print("Skipping plotting due to empty data after filtering/alignment.")

print("Error visualization script finished.")