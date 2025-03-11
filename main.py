from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import io
import uvicorn
import logging
from feature_extraction import *

app = FastAPI()
logger = logging.getLogger('uvicorn.error')

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
    features = {}

    numeric_df = df.select_dtypes(include=["number"])
    features["outliers"] = detect_outliers_zscore(numeric_df)
    features["low_variance_columns"] = detect_low_variance(numeric_df)
    transformed_df, skewed_cols = handle_skewness(numeric_df)
    features["skewed_columns"] = skewed_cols

    features["high_cardinality_columns"] = detect_high_cardinality(df)
    frequent_categories, rare_categories = detect_frequent_and_rare_categories(df)
    features["frequent_categories"] = frequent_categories
    features["rare_categories"] = rare_categories
    features["suggested_encoding"] = suggest_encoding_method(df)

    logger.info('POST /upload')
    return {"filename": file.filename, "features": features}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)