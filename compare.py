# %%
# Script to Evaluate Predictions and Plot Actual vs. Predicted
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys
import warnings

# Ignore specific warnings during plotting if needed
warnings.filterwarnings("ignore", message="Passing `palette` without assigning `hue`")

# --- Configuration ---
# <<< Make sure these point to your actual test data and LATEST prediction file >>>
ACTUAL_TEST_DATA_FILE = 'Timeseries_six_test.csv'
PREDICTION_DATA_FILE = 'predictions.csv' # e.g., predictions from Shifted Output script
OUTPUT_PLOT_FILE = 'evaluation_actual_vs_pred_plot.png' # New plot name

SERIES_TO_PLOT = ['Series1', 'Series2', 'Series3', 'Series4', 'Series5', 'Series6']
PLOT_PERIOD_LIMIT = '7D' # Example: Plot first 7 days of available comparison period
# PLOT_PERIOD_LIMIT = None # Uncomment to plot all available comparison period

# --- Helper Function for sMAPE ---
def symmetric_mean_absolute_percentage_error(y_true, y_pred, epsilon=1e-10):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    denominator = np.maximum(denominator, epsilon) # Avoid division by zero
    smape = np.mean(numerator / denominator) * 100
    return smape

# --- 1. Load Data ---
print(f"Loading actual test data from: {ACTUAL_TEST_DATA_FILE}")
try:
    actual_df_orig = pd.read_csv(ACTUAL_TEST_DATA_FILE, parse_dates=['Date'], index_col='Date')
    if not isinstance(actual_df_orig.index, pd.DatetimeIndex): raise ValueError("Index not DatetimeIndex.")
    if actual_df_orig.empty: raise ValueError("Actual test data file is empty.")
    original_cols = [col for col in actual_df_orig.columns if col.startswith('Series')]
    if not original_cols: raise ValueError("No 'Series' columns found in actual data.")
    actual_df = actual_df_orig[original_cols].copy() # Keep only series columns
except Exception as e:
    print(f"Error loading actual test data: {e}"); sys.exit()

print(f"Loading prediction data from: {PREDICTION_DATA_FILE}")
try:
    pred_df_orig = pd.read_csv(PREDICTION_DATA_FILE, parse_dates=['Date'], index_col='Date')
    if not isinstance(pred_df_orig.index, pd.DatetimeIndex): raise ValueError("Index not DatetimeIndex.")
    if pred_df_orig.empty: raise ValueError("Prediction data file is empty.")
    if not all(col in pred_df_orig.columns for col in original_cols): raise ValueError("Prediction file missing some Series columns.")
    pred_df = pred_df_orig[original_cols].copy() # Keep only series columns and ensure order
except Exception as e:
    print(f"Error loading prediction data: {e}"); sys.exit()

# --- 2. Validate and Align Data ---
print("Validating and aligning data...")
if not actual_df.index.equals(pred_df.index):
    print("Warning: Indices of actual and prediction files do not match exactly.")
    common_index = actual_df.index.intersection(pred_df.index)
    if common_index.empty: raise ValueError("Indices have no overlap. Cannot compare.")
    print(f"Aligning using {len(common_index)} common timestamps...")
    actual_df = actual_df.loc[common_index]
    pred_df = pred_df.loc[common_index]
    if not actual_df.index.equals(pred_df.index): raise ValueError("Alignment failed.")
else:
    print("Indices match.")

if not actual_df.columns.equals(pred_df.columns):
    raise ValueError("Columns do not match between actual and prediction files.")

# --- 3. Calculate Metrics ---
print("\nCalculating evaluation metrics...")
evaluation_results = {}
# Combine for easier NaN handling during metric calculation
eval_combo = pd.concat([actual_df.add_suffix('_actual'), pred_df.add_suffix('_pred')], axis=1)

for col in original_cols:
    actual_col_name = f"{col}_actual"; pred_col_name = f"{col}_pred"
    print(f"-- {col} --")
    # Drop rows where EITHER actual OR prediction is NaN for metric calculation
    eval_subset = eval_combo[[actual_col_name, pred_col_name]].dropna()
    if len(eval_subset) > 0:
        y_test_aligned = eval_subset[actual_col_name]; y_pred_aligned = eval_subset[pred_col_name]
        mse = mean_squared_error(y_test_aligned, y_pred_aligned)
        rmse = np.sqrt(mse); mae = mean_absolute_error(y_test_aligned, y_pred_aligned)
        r2 = r2_score(y_test_aligned, y_pred_aligned)
        smape = symmetric_mean_absolute_percentage_error(y_test_aligned, y_pred_aligned)
        mean_abs_actual = np.mean(np.abs(y_test_aligned)); mae_ratio = (mae / mean_abs_actual) * 100 if mean_abs_actual > 1e-10 else np.inf
        evaluation_results[col] = {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'sMAPE (%)': smape, 'MAE/Mean (%)': mae_ratio}
        print(f'  RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}, sMAPE: {smape:.2f}%, MAE/Mean: {mae_ratio:.2f}%')
    else:
        print("  No overlapping non-NaN data found for metric calculation.")
        evaluation_results[col] = {'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan, 'sMAPE (%)': np.nan, 'MAE/Mean (%)': np.nan}

print("\n--- Final Evaluation Summary ---"); eval_summary = pd.DataFrame(evaluation_results).T
pd.options.display.float_format = '{:.4f}'.format; print(eval_summary); print("-------------------------------")

# --- 4. Prepare Data for Plotting ---
# Use the combined dataframe and drop rows where prediction is NaN
# (This handles the initial gap caused by the shifted prediction)
plot_data_available = eval_combo.dropna(subset=[f"{col}_pred" for col in original_cols])
if plot_data_available.empty:
    print("\nNo data available for plotting after dropping initial prediction NaNs.")
    sys.exit()

# Apply optional time limit for plotting
plot_start_date = plot_data_available.index.min()
if PLOT_PERIOD_LIMIT:
    plot_end_date = plot_start_date + pd.Timedelta(PLOT_PERIOD_LIMIT)
    print(f"\nLimiting plot period to: {plot_start_date} to {plot_end_date}")
    plot_data = plot_data_available.loc[plot_start_date:plot_end_date]
    if plot_data.empty :
         print(f"Warning: No data found in the specified plot limit '{PLOT_PERIOD_LIMIT}'.")
else:
    plot_data = plot_data_available

# --- 5. Create Plots ---
print("Generating Actual vs. Predicted plots...")
valid_plot = not plot_data.empty

if valid_plot:
    n_series = len(original_cols); n_cols = 2; n_rows = (n_series + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4), sharex=True)
    axes = axes.flatten(); plot_successful=False; last_plotted_index = -1

    for i, col in enumerate(original_cols):
        actual_col_name = f"{col}_actual"; pred_col_name = f"{col}_pred"
        if actual_col_name not in plot_data or pred_col_name not in plot_data or \
           plot_data[actual_col_name].empty or plot_data[pred_col_name].empty:
             print(f"Skipping plot for {col}, insufficient data in period.")
             if i < len(axes) and axes[i].figure == fig: fig.delaxes(axes[i])
             continue

        ax = axes[i]
        # Plot actual values
        plot_data[actual_col_name].plot(ax=ax, label='Actual', color='blue', linewidth=1.0, alpha=0.8)
        # Plot predictions
        plot_data[pred_col_name].plot(ax=ax, label='Predicted', color='red', linewidth=1.0, alpha=0.8)

        ax.set_title(f'{col} - Actual vs. Predicted')
        ax.set_ylabel('Value')
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
        if j < len(axes) and hasattr(axes[j], 'figure') and axes[j].figure == fig: fig.delaxes(axes[j])

    fig.suptitle('Actual vs. Predicted Values Over Time (Test Period)', fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])

    try: plt.savefig(OUTPUT_PLOT_FILE); print(f"Plots saved to {OUTPUT_PLOT_FILE}")
    except Exception as e: print(f"Error saving plot: {e}")
    # plt.show()
else:
     print("Skipping plotting - no valid data after alignment and NaN handling.")

print("\nEvaluation and plotting script finished.")
# %%