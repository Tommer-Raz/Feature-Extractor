from fastapi import FastAPI, UploadFile, File
import pandas as pd
import io
from feature_extraction import *
app = FastAPI()

@app.post("/upload/")
async def extract_features(file: UploadFile = File(...)):
    # Read the CSV file
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

    # Feature extraction
    features = {
        "column_types": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "unique_values": df.nunique().to_dict(),
    }

    # Statistical summary for numerical columns
    # numeric_df = df.select_dtypes(include=["number"])
    # if not numeric_df.empty:
    #     stats = numeric_df.describe().to_dict()
    #     features["statistics"] = stats

    numeric_df = df.select_dtypes(include=["number"])
    features["outliers"] = detect_outliers_zscore(numeric_df)
    features["low_variance_columns"] = detect_low_variance(numeric_df)
    transformed_df, skewed_cols = handle_skewness(numeric_df)
    features["skewed_columns"] = skewed_cols
    
    features["high_cardinality_columns"] = detect_high_cardinality(df)
    frequent_categories, rare_categories = detect_frequent_and_rare_categories(df)
    features["frequent_categories"] = frequent_categories
    features["rare_categories"] = rare_categories
    return {"filename": file.filename, "features": features}