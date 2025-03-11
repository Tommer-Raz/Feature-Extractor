from scipy.stats import zscore, skew
import numpy as np

def detect_outliers_zscore(df, threshold=3):
    """Detect outliers using Z-score method."""
    outliers = {}

    for col in df.columns:
        clean_data = df[col].dropna() # remove nan values as they are not part of the zscore
        z_scores = np.abs(zscore(clean_data))  # Calculate Z-scores
        outlier_values = clean_data[z_scores > threshold].tolist()
        outliers[col] = outlier_values

    return outliers

def detect_low_variance(df, threshold=0.01):
    """Identify columns with low variance."""
    variances = df.var()
    low_variance_cols = [col for col in variances.index if variances[col] < threshold]
    return low_variance_cols

def handle_skewness(df):
    """Apply log transformation to highly skewed columns."""
    skewness = df.apply(lambda x: skew(x.dropna()))  # Exclude NaNs
    skewed_cols = skewness[abs(skewness) > 1].index.tolist()

    transformed_df = df.copy()
    for col in skewed_cols:
        transformed_df[col] = np.log1p(df[col])  # log1p avoids log(0) issues
    
    return transformed_df, skewed_cols

def detect_high_cardinality(df, threshold=50, relative_threshold=0.1):
    """Detect high-cardinality categorical columns."""
    categorical_df = df.select_dtypes(include=["object", "category"])
    high_card_cols = []
    
    total_rows = len(df)
    for col in categorical_df.columns:
        unique_count = df[col].nunique()
        
        # High cardinality if unique values exceed absolute threshold OR relative threshold
        if unique_count > threshold or unique_count / total_rows > relative_threshold:
            high_card_cols.append(col)

    return high_card_cols

def detect_frequent_and_rare_categories(df, freq_threshold=0.05, rare_threshold=0.01):
    """Detect frequent and rare categories in categorical columns."""
    categorical_df = df.select_dtypes(include=["object", "category"])
    frequent_categories = {}
    rare_categories = {}
    
    for col in categorical_df.columns:
        value_counts = df[col].value_counts(normalize=True)  # Get percentage of each category
        frequent = value_counts[value_counts > freq_threshold].index.tolist()
        rare = value_counts[value_counts < rare_threshold].index.tolist()

        frequent_categories[col] = frequent
        rare_categories[col] = rare

    return frequent_categories, rare_categories

def suggest_encoding_method(df):
    """Suggest encoding methods based on patterns in categorical columns."""
    encoding_suggestions = {}
    
    categorical_df = df.select_dtypes(include=["object", "category"])
    high_cardinality_cols = detect_high_cardinality(df)
    for col in categorical_df.columns:
        # 1. High Cardinality
        if col in high_cardinality_cols:
            encoding_suggestions[col] = "Target Encoding or Frequency Encoding"

        else:
            encoding_suggestions[col] = "One-Hot Encoding"
    
    return encoding_suggestions