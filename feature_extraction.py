from scipy.stats import zscore
import numpy as np

def detect_outliers_zscore(df, threshold=3):
    """Detect outliers using Z-score method."""
    outliers = {}

    for col in df.columns:
        clean_data = df[col].dropna()
        z_scores = np.abs(zscore(clean_data))  # Calculate Z-scores
        outlier_values = clean_data[z_scores > threshold].tolist()
        outliers[col] = outlier_values

    return outliers

def detect_low_variance(df, threshold=0.01):
    """Identify columns with low variance."""
    variances = df.var()
    low_variance_cols = [col for col in variances.index if variances[col] < threshold]
    return low_variance_cols

