import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from tqdm import tqdm
import sys
import warnings

# Ignore specific warnings during plotting if needed
warnings.filterwarnings("ignore", message="Passing `palette` without assigning `hue`")

# --- Configuration ---
# <<< Make sure this points to your LATEST prediction file >>>
PREDICTIONS_FILE = 'predictions_abs_trig_Xvol_gapped_cv.csv'
RAW_DATA_FILE = 'data.csv'
OUTPUT_PLOT_ERR_TIME = 'error_analysis_time_plot.png'
OUTPUT_PLOT_ERR_DIST = 'error_analysis_dist_plot.png'
OUTPUT_PLOT_ERR_ACF = 'error_analysis_acf_plot.png'

SERIES_TO_PLOT = ['Series1', 'Series2', 'Series3', 'Series4', 'Series5', 'Series6']
PLOT_PERIOD_LIMIT = '14D' # Limit time plot duration
# PLOT_PERIOD_LIMIT = None
IQR_MULTIPLIER = 1.5
ACF_LAGS = 60 * 4 # Show ACF for up to 4 hours of lags

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

# --- Align Data & Calculate Errors ---
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
else: print("Indices successfully aligned.")

errors = actual_test_period - predictions_filtered
print(f"Calculated errors for {len(errors)} time points.")

# Apply plot period limit for time plot
plot_start_date = errors.index.min()
if PLOT_PERIOD_LIMIT:
    plot_end_date = plot_start_date + pd.Timedelta(PLOT_PERIOD_LIMIT)
    print(f"Limiting time plot period to: {plot_start_date} to {plot_end_date}")
    errors_time_plot_data = errors.loc[plot_start_date:plot_end_date]
else:
    errors_time_plot_data = errors

# --- Plot Errors Over Time ---
print("Generating error time plots...")
valid_time_plot = not errors_time_plot_data.empty
if valid_time_plot:
    n = len(common_cols); n_cols = 2; n_rows = (n + n_cols - 1) // n_cols
    fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 3.5), sharex=True)
    axes1 = axes1.flatten(); plot_successful=False; last_plotted_index = -1
    for i, col in enumerate(common_cols):
        if col not in errors_time_plot_data or errors_time_plot_data[col].empty: continue
        ax = axes1[i]; errors_time_plot_data[col].plot(ax=ax, label='Error', color='green', lw=1); ax.axhline(0, color='red', ls='--', lw=0.8, label='Zero');
        ax.set_title(f'{col} Error Over Time'); ax.set_ylabel('Actual - Predicted'); ax.legend(); ax.grid(True, ls='--', alpha=0.6); plot_successful=True; last_plotted_index=i
    if plot_successful: fig1.autofmt_xdate(rotation=30, ha='right')
    for j in range(last_plotted_index + 1, len(axes1)): fig1.delaxes(axes1[j])
    fig1.suptitle('Prediction Errors (Actual - Predicted) Over Time', fontsize=16, y=1.02); plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    try: plt.savefig(OUTPUT_PLOT_ERR_TIME); print(f"Error time plots saved to {OUTPUT_PLOT_ERR_TIME}")
    except Exception as e: print(f"Error saving plot: {e}")
    plt.close(fig1)
else: print("Skipping error time plot - no data.")

# --- Plot Error Distribution ---
print("\nGenerating error distribution plots...")
valid_dist_plot = not errors.empty
if valid_dist_plot:
    n = len(common_cols); n_cols = 2; n_rows = (n + n_cols - 1) // n_cols
    fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 4))
    axes2 = axes2.flatten(); plot_successful=False; last_plotted_index = -1
    for i, col in enumerate(common_cols):
        if col not in errors or errors[col].empty: continue
        ax = axes2[i]; sns.histplot(errors[col], kde=True, ax=ax, bins=50); error_mean = errors[col].mean(); error_std = errors[col].std()
        ax.axvline(error_mean, color='r', linestyle='--', label=f'Mean: {error_mean:.2f}')
        ax.set_title(f'{col} Error Distribution (Std: {error_std:.2f})'); ax.set_xlabel('Error'); ax.legend(); ax.grid(True, ls='--', alpha=0.6); plot_successful=True; last_plotted_index=i
    if plot_successful: fig2.suptitle('Distribution of Prediction Errors', fontsize=16, y=1.02); plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    for j in range(last_plotted_index + 1, len(axes2)): fig2.delaxes(axes2[j])
    try: plt.savefig(OUTPUT_PLOT_ERR_DIST); print(f"Error distribution plots saved to {OUTPUT_PLOT_ERR_DIST}")
    except Exception as e: print(f"Error saving plot: {e}")
    plt.close(fig2)
else: print("Skipping error distribution plot - no data.")

# --- Plot Error ACF ---
print("\nGenerating error ACF plots...")
valid_acf_plot = not errors.empty
if valid_acf_plot:
    n = len(common_cols); n_cols = 2; n_rows = (n + n_cols - 1) // n_cols
    fig3, axes3 = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 3.5))
    axes3 = axes3.flatten(); plot_successful=False; last_plotted_index = -1
    for i, col in enumerate(common_cols):
        if col not in errors or errors[col].isna().all(): continue # Skip if all NaN
        ax = axes3[i]
        try: # ACF plot can fail if variance is zero etc.
            sm.graphics.tsa.plot_acf(errors[col].dropna(), lags=ACF_LAGS, ax=ax, title=f'{col} Error Autocorrelation')
            ax.grid(True, ls='--', alpha=0.6)
            plot_successful=True; last_plotted_index=i
        except Exception as e:
            print(f"Could not generate ACF for {col}: {e}")
            fig3.delaxes(ax) # Remove axis if plot failed

    if plot_successful: fig3.suptitle('Autocorrelation of Prediction Errors', fontsize=16, y=1.02); plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    for j in range(last_plotted_index + 1, len(axes3)):
         if j < len(axes3) and axes3[j].figure == fig3: fig3.delaxes(axes3[j])
    try: plt.savefig(OUTPUT_PLOT_ERR_ACF); print(f"Error ACF plots saved to {OUTPUT_PLOT_ERR_ACF}")
    except Exception as e: print(f"Error saving plot: {e}")
    plt.close(fig3)
else: print("Skipping error ACF plot - no data.")


print("\nError analysis finished.")