from scipy.stats import zscore
import numpy as np

def detect_outliers_zscore(df, threshold=3):
    """Detect outliers using Z-score method."""
    outliers = {}
    numeric_df = df.select_dtypes(include=["number"])

    for col in numeric_df.columns:
        clean_data = numeric_df[col].dropna()
        z_scores = np.abs(zscore(clean_data))  # Calculate Z-scores
        outlier_values = clean_data[z_scores > threshold].tolist()
        outliers[col] = outlier_values

    return outliers

