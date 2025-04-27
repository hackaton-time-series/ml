import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import sys

# --- Configuration ---
# <<< Make sure this points to your LATEST prediction file >>>
PREDICTIONS_FILE = 'predictions_abs_trig_features_gapped_cv.csv'
RAW_DATA_FILE = 'data.csv'
OUTPUT_PLOT_FILE = 'prediction_scatter_plots.png' # Output plot name

SERIES_TO_PLOT = ['Series1', 'Series2', 'Series3', 'Series4', 'Series5', 'Series6']
IQR_MULTIPLIER = 1.5
# Sample data for scatter plot? Plotting all points (~93k) can be very slow and create dense blobs.
# Set SAMPLE_SIZE to None to plot all, or an integer (e.g., 10000) to plot a random sample.
SAMPLE_SIZE = 10000

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

# Combine actual and predicted for easier plotting/sampling
comparison_df = pd.concat([actual_test_period, predictions_filtered.add_suffix('_pred')], axis=1)

# Sample data if requested
if SAMPLE_SIZE and SAMPLE_SIZE < len(comparison_df):
    print(f"Sampling {SAMPLE_SIZE} points for scatter plot...")
    plot_data = comparison_df.sample(n=SAMPLE_SIZE, random_state=42)
else:
    print("Using all available points for scatter plot.")
    plot_data = comparison_df


# --- Create Plots ---
print("Generating scatter plots...")
valid_plot = not plot_data.empty

if valid_plot:
    n_series = len(common_cols); n_cols = 2; n_rows = (n_series + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 5)) # Adjusted size
    axes = axes.flatten()
    plot_successful = False; last_plotted_index = -1

    for i, col in enumerate(common_cols):
        actual_col = col
        pred_col = f'{col}_pred'

        if actual_col not in plot_data or pred_col not in plot_data or \
           plot_data[actual_col].empty or plot_data[pred_col].empty:
             print(f"Skipping plot for {col}, insufficient data.")
             if i < len(axes) and axes[i].figure == fig: fig.delaxes(axes[i])
             continue

        ax = axes[i]
        # Use seaborn's scatterplot for potentially better handling of overlaps (alpha)
        sns.scatterplot(x=actual_col, y=pred_col, data=plot_data, ax=ax, alpha=0.3, s=10) # s adjusts point size

        # Add y=x line (perfect prediction)
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]
        ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label='y=x (Perfect)') # Red dashed line
        ax.set_aspect('equal', adjustable='box') # Make axes equal scale
        ax.set_xlim(lims); ax.set_ylim(lims) # Apply limits

        ax.set_title(f'{col}: Predicted vs. Actual')
        ax.set_xlabel('Actual Value')
        ax.set_ylabel('Predicted Value')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        plot_successful = True
        last_plotted_index = i

    # Hide any unused subplots
    for j in range(last_plotted_index + 1, len(axes)):
         if j < len(axes) and axes[j].figure == fig: fig.delaxes(axes[j])

    fig.suptitle('Predicted vs. Actual Scatter Plots (Test Period Sample)', fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])

    try: plt.savefig(OUTPUT_PLOT_FILE); print(f"Scatter plots saved to {OUTPUT_PLOT_FILE}")
    except Exception as e: print(f"Error saving plot: {e}")
    # plt.show()
else:
     print("Skipping plotting due to empty data after filtering/alignment.")

print("Scatter plot visualization script finished.")