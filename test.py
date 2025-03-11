import pytest
from fastapi.testclient import TestClient
from main import app  # Import FastAPI app from your main script
import io
import pandas as pd

client = TestClient(app)

# Sample CSV content for testing
CSV_CONTENT = """date,category,value
2023-01-01,A,10
2023-01-02,B,20
2023-01-03,C,30
2023-01-04,A,40
2023-01-05,B,50
"""

INVALID_CSV_CONTENT = """datecategory,value
2023-01-01,A,10,
2023-01-02,B,20,
2023-01-0
2023-01-04,A,40,,,,,,,,,,,
202
"""

@pytest.fixture
def valid_csv():
    """Creates a valid CSV file for testing."""
    return io.BytesIO(CSV_CONTENT.encode())

@pytest.fixture
def invalid_csv():
    """Creates an invalid CSV file (wrong structure)."""
    return io.BytesIO(INVALID_CSV_CONTENT.encode())

@pytest.fixture
def non_csv_file():
    """Creates a non-CSV file (e.g., .txt)."""
    return io.BytesIO(b"This is a text file, not a CSV.")


def test_upload_valid_csv(valid_csv):
    response = client.post("/upload/", files={"file": ("data.csv", valid_csv, "text/csv")})
    assert response.status_code == 200

    json_data = response.json()
    assert "features" in json_data

    features = json_data["features"]
    assert "high_cardinality" in features
    assert "suggested_encoding" in features
    assert "outliers" in features
    assert "skewed_columns" in features
    assert "rare_categories" in features

def test_upload_non_csv(non_csv_file):
    response = client.post("/upload/", files={"file": ("data.txt", non_csv_file, "text/plain")})
    assert response.status_code == 400
    assert "Invalid file type" in response.json()["detail"]

def test_upload_invalid_csv(invalid_csv):
    response = client.post("/upload/", files={"file": ("invalid.csv", invalid_csv, "text/csv")})
    assert response.status_code == 400
    assert "Error reading CSV file" in response.json()["detail"]

def test_upload_no_file():
    response = client.post("/upload/")
    assert response.status_code == 422  # FastAPI will return a 422 error for missing required fields

def test_high_cardinality():
    df = pd.DataFrame({
        "category": [f"item_{i}" for i in range(100)],  # 100 unique categories
        "value": range(100)
    })
    csv_data = df.to_csv(index=False).encode("utf-8")
    response = client.post("/upload/", files={"file": ("high_cardinality.csv", io.BytesIO(csv_data), "text/csv")})
    
    assert response.status_code == 200
    json_data = response.json()
    assert "category" in json_data["features"]["high_cardinality"]