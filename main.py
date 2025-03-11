from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import io
import uvicorn
import logging
from feature_extraction import *

app = FastAPI()
logger = logging.getLogger('uvicorn.error')

def extract_numeric_features(df):
    features = {}

    features["outliers"] = detect_outliers_zscore(df)
    features["low_variance_columns"] = detect_low_variance(df)
    transformed_df, skewed_cols = handle_skewness(df)
    features["skewed_columns"] = skewed_cols

    return features, transformed_df

def extract_categorial_features(df):
    features = {}

    features["high_cardinality"] = detect_high_cardinality(df)
    frequent_categories, rare_categories = detect_frequent_and_rare_categories(df)
    features["frequent_categories"] = frequent_categories
    features["rare_categories"] = rare_categories
    features["suggested_encoding"] = suggest_encoding_method(df)

    return features

@app.post("/upload/")
async def extract_features(file: UploadFile = File(...)):
    # Read the CSV file
    contents = await file.read()
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV file.")
    
    try:
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    except Exception as e:
        raise HTTPException(status_code=400, 
                            detail=f"Error reading CSV file, make sure this a csv file: {str(e)}")
    
    if df.empty:
        raise HTTPException(status_code=400, detail="The uploaded CSV file is empty.")

    if df.columns.duplicated().any():
        raise HTTPException(status_code=400, detail="CSV file contains duplicate column names.")
    
    numeric_features, non_skew_df = extract_numeric_features(df)
    categorial_features = extract_categorial_features(df)

    features = {**numeric_features, **categorial_features}
    return {"filename": file.filename, "features": features}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)