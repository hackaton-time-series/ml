import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno # Library specifically for visualizing missing data
import sys

# Configuration
RAW_DATA_FILE = 'data.csv'

print(f"Python version: {sys.version}")
print(f"Pandas version: {pd.__version__}")
print(f"Matplotlib version: {plt.matplotlib.__version__}")
print(f"Missingno version: {msno.__version__}")


# --- 1. Load Raw Data ---
print(f"\nLoading raw data from {RAW_DATA_FILE}...")
try:
    data = pd.read_csv(RAW_DATA_FILE, parse_dates=['Date'], index_col='Date')
    original_cols = [col for col in data.columns if col.startswith('Series')]
    print(f"Raw data shape: {data.shape}")
    if data.empty: print("Error: Data file loaded as empty DataFrame."); exit()
    if not isinstance(data.index, pd.DatetimeIndex): print("Error: Index is not a DatetimeIndex."); exit()
    print(f"Time range: {data.index.min()} to {data.index.max()}")
    print(f"Columns loaded: {data.columns.tolist()}")
except Exception as e:
    print(f"Error loading or parsing data: {e}"); exit()

# --- 2. Calculate Missing Data Percentage ---
print("\n--- Missing Data Analysis ---")
if not all(col in data.columns for col in original_cols):
     print("Warning: Not all expected 'SeriesX' columns found.")
     original_cols = [col for col in original_cols if col in data.columns]

if not original_cols: print("Error: No 'SeriesX' columns found."); exit()

# <<< FIX: Replace np.product with pandas .size attribute >>>
# total_cells = np.product(data[original_cols].shape) # Old line causing error
total_cells = data[original_cols].size                 # Corrected line

total_missing = data[original_cols].isna().sum().sum()

if total_cells == 0: print("Error: DataFrame has zero cells."); exit()

percent_missing_total = (total_missing / total_cells) * 100
print(f"Total missing cells in Series columns: {total_missing}")
print(f"Total cells in Series columns: {total_cells}")
print(f"Overall percentage missing in Series columns: {percent_missing_total:.2f}%")

print("\nMissing data percentage per series:")
missing_per_series = data[original_cols].isna().mean() * 100
print(missing_per_series.round(2))

# --- 3. Visualize Missing Data Patterns ---
print("\nGenerating missing data visualizations (this might take a moment)...")
FIGSIZE_MATRIX = (18, 6); FIGSIZE_HEATMAP = (8, 7); FIGSIZE_BAR = (10, 5)

# 3a: Matrix plot
print("Generating matrix plot...")
try:
    fig_matrix = msno.matrix(data[original_cols], freq='D', figsize=FIGSIZE_MATRIX, sparkline=False)
    plt.title('Missing Data Matrix (Daily Aggregation)', fontsize=14)
    try: plt.tight_layout()
    except Exception: print("Note: tight_layout failed on matrix plot.")
    plt.savefig('missing_data_matrix.png')
    print("Matrix plot saved to missing_data_matrix.png")
    plt.close()
except Exception as e: print(f"Could not generate matrix plot: {e}")

# 3b: Heatmap
print("\nGenerating missingness heatmap...")
try:
    fig_heatmap = msno.heatmap(data[original_cols], figsize=FIGSIZE_HEATMAP)
    plt.title('Missingness Correlation Heatmap', fontsize=14)
    try: plt.tight_layout()
    except Exception: print("Note: tight_layout failed on heatmap.")
    plt.savefig('missing_data_heatmap.png')
    print("Heatmap saved to missing_data_heatmap.png")
    plt.close()
except Exception as e: print(f"Could not generate heatmap: {e}")

# 3c: Bar chart
print("\nGenerating missingness bar chart...")
try:
    fig_bar = msno.bar(data[original_cols], figsize=FIGSIZE_BAR)
    plt.title('Data Presence per Series', fontsize=14)
    try: plt.tight_layout()
    except Exception: print("Note: tight_layout failed on bar chart.")
    plt.savefig('missing_data_bar.png')
    print("Bar chart saved to missing_data_bar.png")
    plt.close()
except Exception as e: print(f"Could not generate bar chart: {e}")

print("\nMissing data analysis finished.")