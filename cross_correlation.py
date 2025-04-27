import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm # For CCF calculation
from tqdm import tqdm
import sys

# Configuration
RAW_DATA_FILE = 'data.csv'
# Use the same cleaning as the main script (IQR) before CCF
IQR_MULTIPLIER = 1.5
MAX_LAG = 60 * 6 # Calculate for +/- 6 hours lag (adjust if needed)
OUTPUT_PLOT_FILE = 'cross_correlation_plots.png' # Define output file name

print(f"Python version: {sys.version}")
print(f"Pandas version: {pd.__version__}")
print(f"Matplotlib version: {plt.matplotlib.__version__}")
try:
    import statsmodels
    print(f"Statsmodels version: {statsmodels.__version__}")
except ImportError:
    print("Error: statsmodels not found. Please install it: pip install statsmodels")
    exit()


# --- 1. Load and Clean Data ---
print(f"Loading raw data from {RAW_DATA_FILE}...")
try:
    data = pd.read_csv(RAW_DATA_FILE, parse_dates=['Date'], index_col='Date')
    original_cols = [col for col in data.columns if col.startswith('Series')]
    print(f"Raw data shape: {data.shape}")
    if data.empty: print("Error: Data file empty."); exit()
    if not isinstance(data.index, pd.DatetimeIndex): print("Error: Index not DatetimeIndex."); exit()
except Exception as e: print(f"Error loading data: {e}"); exit()

print("Cleaning data (IQR method)...")
# Initial interpolate
data[original_cols] = data[original_cols].interpolate(method='time')
# IQR Outlier Masking
for col in tqdm(original_cols, desc="Cleaning (IQR)", disable=True):
    Q1 = data[col].quantile(0.25); Q3 = data[col].quantile(0.75); IQR = Q3 - Q1
    lower_bound = Q1 - IQR_MULTIPLIER * IQR; upper_bound = Q3 + IQR_MULTIPLIER * IQR
    is_outlier = (data[col] < lower_bound) | (data[col] > upper_bound)
    data[col] = data[col].mask(is_outlier)
# Interpolate Outlier Gaps
data[original_cols] = data[original_cols].interpolate(method='time')
data.dropna(subset=original_cols, inplace=True)
# Ensure Frequency
inferred_freq = pd.infer_freq(data.index); freq_to_use = 'min'
if inferred_freq:
    if 'MIN' in inferred_freq.upper() or inferred_freq.upper() == 'T': freq_to_use = 'min'
    else: freq_to_use = inferred_freq
data = data.asfreq(freq_to_use)
data[original_cols] = data[original_cols].interpolate(method='time')
data.dropna(subset=original_cols, inplace=True)
print(f"Cleaned data shape: {data.shape}")
if data.empty: print("Error: Data empty after cleaning."); exit()

# --- 2. Calculate and Plot Cross-Correlations ---
print(f"\nCalculating Cross-Correlations (+/- {MAX_LAG} minutes lag)...")

df_to_use = data # Use original frequency

target_series = 'Series1'
other_series = [col for col in original_cols if col != target_series]

n_series = len(other_series)
n_cols = 2
n_rows = (n_series + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 3.5), sharey=True)
axes = axes.flatten()
plot_successful = False
last_plotted_index = -1

if target_series not in df_to_use.columns:
     print(f"Error: Target series '{target_series}' not found.")
     exit()

for i, other_col in enumerate(other_series):
    if other_col not in df_to_use.columns:
         print(f"Skipping CCF for {other_col}, not found.")
         if i < len(axes) and axes[i].figure == fig: fig.delaxes(axes[i])
         continue

    print(f"  Calculating CCF between {target_series} and {other_col}...")
    try:
        # Ensure data is float and has no NaNs for ccf
        series_x = df_to_use[other_col].astype(float).dropna()
        series_y = df_to_use[target_series].astype(float).dropna()
        # Align indices before passing to ccf
        common_idx = series_x.index.intersection(series_y.index)
        if len(common_idx) < 2: # Need at least 2 points for correlation
             print(f"Warning: Not enough overlapping data for {other_col}. Skipping.")
             if i < len(axes) and axes[i].figure == fig: fig.delaxes(axes[i])
             continue
        series_x = series_x.loc[common_idx]
        series_y = series_y.loc[common_idx]


        ccf_pos_lags = sm.tsa.stattools.ccf(series_x, series_y, adjusted=False)
        ccf_neg_lags = sm.tsa.stattools.ccf(series_y, series_x, adjusted=False)

        # Combine positive and negative lags up to MAX_LAG
        max_calc_lag = len(ccf_pos_lags) - 1 # Max lag actually calculated by statsmodels
        current_max_lag = min(MAX_LAG, max_calc_lag) # Use the smaller of desired or possible

        lags = np.arange(-current_max_lag, current_max_lag + 1)
        corr = np.concatenate([ccf_neg_lags[current_max_lag:0:-1], ccf_pos_lags[:current_max_lag+1]])

        if len(corr) != len(lags):
            print(f"Warning: Length mismatch for {other_col}. Corr len: {len(corr)}, Lags len: {len(lags)}. Skipping plot.")
            if i < len(axes) and axes[i].figure == fig: fig.delaxes(axes[i])
            continue

    except Exception as e:
        print(f"Error calculating CCF for {other_col}: {e}")
        if i < len(axes) and axes[i].figure == fig: fig.delaxes(axes[i])
        continue

    ax = axes[i]
    # <<< FIX: Removed use_line_collection=True >>>
    ax.stem(lags, corr, markerfmt=' ', basefmt="k-", linefmt='b-')
    # <<< End Fix >>>

    ax.set_title(f'CCF: {target_series} vs {other_col}')
    ax.set_xlabel('Lag (minutes)')
    ax.set_ylabel('Correlation')
    ax.grid(True, linestyle='--', alpha=0.6)

    # Find peak correlation within the plotted range
    if len(corr) > 0:
        peak_idx = np.argmax(np.abs(corr))
        peak_lag = lags[peak_idx]
        peak_corr = corr[peak_idx] # Use actual corr value, not abs
        ax.annotate(f'Peak @ lag {peak_lag}\nCorr ~ {peak_corr:.2f}', xy=(peak_lag, peak_corr),
                    xytext=(peak_lag, peak_corr + np.sign(peak_corr)*0.1 if peak_corr != 0 else 0.1 ), # Adjust text position
                    ha='center',
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))
    else:
         print(f"Warning: Correlation array empty for {other_col}")


    plot_successful = True
    last_plotted_index = i

# Hide any unused subplots
for j in range(last_plotted_index + 1, len(axes)):
     if j < len(axes) and axes[j].figure == fig: fig.delaxes(axes[j])

if plot_successful:
    fig.suptitle(f'Cross-Correlation Functions relative to {target_series} (Lag=Minutes)', fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(OUTPUT_PLOT_FILE)
    print(f"\nCross-correlation plots saved to {OUTPUT_PLOT_FILE}")
    # plt.show()
else:
    print("\nNo cross-correlation plots were generated.")

print("\nCross-correlation analysis finished.")