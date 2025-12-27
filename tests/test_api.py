import pytest
from fastapi import status

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True

def test_model_info(client):
    response = client.get("/model/info")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "model_type" in data
    assert "training_date" in data
    assert "version" in data

def test_features_endpoint(client):
    response = client.get("/features")
    assert response.status_code == status.HTTP_200_OK
    features = response.json()
    assert isinstance(features, list)
    assert len(features) > 0
    assert all("name" in f and "description" in f for f in features)

def test_predict_endpoint(client):
    test_data = {
        "age": 45,
        "sex": "M",
        "weight_kg": 75.5,
        "baseline_severity": 7,
        "biomarker_day0": 10.2,
        "biomarker_day1": 9.8,
        "biomarker_day2": 9.0,
        "biomarker_day3": 8.2,
        "biomarker_day4": 7.5,
        "biomarker_day5": 6.8
    }
    response = client.post("/predict", json=test_data)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "prediction" in data
    assert "probability_of_response" in data
    assert "dose_intensity" in data
    assert "alert_clinician" in data
    assert "message" in data

def test_batch_predict_endpoint(client, tmp_path):
    # Create a test CSV file
    import pandas as pd
    test_data = {
        "age": [45, 50],
        "sex": ["M", "F"],
        "weight_kg": [75.5, 65.0],
        "baseline_severity": [7, 6],
        "biomarker_day0": [10.2, 9.5],
        "biomarker_day1": [9.8, 9.6],
        "biomarker_day2": [9.0, 9.7],
        "biomarker_day3": [8.2, 9.8],
        "biomarker_day4": [7.5, 9.9],
        "biomarker_day5": [6.8, 10.0]
    }
    df = pd.DataFrame(test_data)
    test_file = tmp_path / "test_batch.csv"
    df.to_csv(test_file, index=False)
    
    with open(test_file, "rb") as f:
        response = client.post(
            "/predict_batch",
            files={"file": ("test_batch.csv", f, "text/csv")}
        )
    
    assert response.status_code == status.HTTP_200_OK
    results = response.json()
    assert isinstance(results, list)
    assert len(results) == 2
    assert all("patient_id" in r for r in results)
    assert all("prediction" in r for r in results)

def test_invalid_batch_predict(client, tmp_path):
    # Test with invalid file type
    test_file = tmp_path / "test.txt"
    test_file.write_text("This is not a CSV file")
    
    with open(test_file, "rb") as f:
        response = client.post(
            "/predict_batch",
            files={"file": ("test.txt", f, "text/plain")}
        )
    
    assert response.status_code == status.HTTP_400_BAD_REQUEST

def test_missing_columns_batch_predict(client, tmp_path):
    # Test with missing required columns
    import pandas as pd
    test_data = {"age": [45], "sex": ["M"]}  # Missing other required columns
    df = pd.DataFrame(test_data)
    test_file = tmp_path / "invalid_batch.csv"
    df.to_csv(test_file, index=False)
    
    with open(test_file, "rb") as f:
        response = client.post(
            "/predict_batch",
            files={"file": ("invalid_batch.csv", f, "text/csv")}
        )
    
    assert response.status_code == status.HTTP_400_BAD_REQUEST