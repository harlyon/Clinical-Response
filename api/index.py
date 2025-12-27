from fastapi import UploadFile, File
import io
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI(title="Clinical Early Response Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    BASE_DIR = Path(__file__).resolve().parent.parent
    model_path = os.path.join(BASE_DIR, "models", "clinical_response_model.pkl")
    scaler_path = os.path.join(BASE_DIR, "models", "clinical_scaler.pkl")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    model_loaded = True

    # Get model's last modified time
    model_mtime = os.path.getmtime(model_path)
    training_date = datetime.fromtimestamp(model_mtime).strftime('%Y-%m-%d')
    
    # Get model metadata if available
    model_info = {
        "model_type": str(type(model).__name__),
        "training_date": training_date,
        "features_used": [
            "age", "sex", "weight_kg", "baseline_severity",
            "biomarker_day0", "biomarker_day1", "biomarker_day2",
            "biomarker_day3", "biomarker_day4", "biomarker_day5"
        ],
        "version": "1.0.0"
    }
except Exception as e:
    model_loaded = False
    print(f"Error loading model or scaler: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify the API is running and models are loaded
    """
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Service Unavailable - Model or scaler not loaded"
        )
    
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "timestamp": datetime.utcnow().isoformat(),
        "api_version": "1.0.0"
    }

class BatchPredictionResponse(BaseModel):
    patient_id: str
    prediction: str
    probability_of_response: float
    dose_intensity: float
    alert_clinician: bool
    message: str

class PatientData(BaseModel):
    age: int
    sex: str 
    weight_kg: float 
    baseline_severity: int
    biomarker_day0: float
    biomarker_day1: float
    biomarker_day2: float
    biomarker_day3: float
    biomarker_day4: float
    biomarker_day5: float

@app.get("/model/info", response_model=Dict[str, Any])
async def get_model_info():
    """
    Returns metadata about the loaded model
    """
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Service Unavailable - Model not loaded"
        )
    return model_info

@app.get("/features", response_model=List[Dict[str, str]])
async def get_feature_descriptions():
    """
    Returns descriptions of all input features
    """
    features = [
        {"name": "age", "type": "int", "description": "Patient age in years"},
        {"name": "sex", "type": "str", "description": "Patient's biological sex (M/F)"},
        {"name": "weight_kg", "type": "float", "description": "Patient weight in kilograms"},
        {"name": "baseline_severity", "type": "int", "description": "Baseline severity score (1-10)"},
        {"name": "biomarker_day0", "type": "float", "description": "Biomarker level at day 0"},
        {"name": "biomarker_day1", "type": "float", "description": "Biomarker level at day 1"},
        {"name": "biomarker_day2", "type": "float", "description": "Biomarker level at day 2"},
        {"name": "biomarker_day3", "type": "float", "description": "Biomarker level at day 3"},
        {"name": "biomarker_day4", "type": "float", "description": "Biomarker level at day 4"},
        {"name": "biomarker_day5", "type": "float", "description": "Biomarker level at day 5"}
    ]
    return features

class PredictionResponse(BaseModel):
    prediction: str
    probability_of_response: float
    dose_intensity: float
    alert_clinician: bool
    message: str
    
    class Config:
        schema_extra = {
            "example": {
                "prediction": "Responder",
                "probability_of_response": 0.87,
                "dose_intensity": 6.62,
                "alert_clinician": False,
                "message": "Patient is likely to respond to treatment."
            }
        }

@app.post("/predict", response_model=PredictionResponse)
async def predict_response(data: PatientData):
    try:
        # 1. Convert Input to DataFrame
        input_dict = data.dict()
        df = pd.DataFrame([input_dict])
        
        # Standardize column names to match model's expected format
        column_mapping = {
            'age': 'Age',
            'sex': 'Sex',
            'weight_kg': 'Weight_Kg',
            'baseline_severity': 'Baseline_Severity',
            'biomarker_day0': 'Biomarker_Day0',
            'biomarker_day1': 'Biomarker_Day1',
            'biomarker_day2': 'Biomarker_Day2',
            'biomarker_day3': 'Biomarker_Day3',
            'biomarker_day4': 'Biomarker_Day4',
            'biomarker_day5': 'Biomarker_Day5'
        }
        df = df.rename(columns=column_mapping)

        df['Dose_Per_Kg'] = 500 / df['Weight_Kg']
        

        df['Total_Change'] = df['Biomarker_Day5'] - df['Biomarker_Day0']
        df['Pct_Change'] = (df['Biomarker_Day5'] - df['Biomarker_Day0']) / (df['Biomarker_Day0'] + 1e-6)
        df['Early_Change'] = df['Biomarker_Day2'] - df['Biomarker_Day0']
        df['Late_Change'] = df['Biomarker_Day5'] - df['Biomarker_Day3']
        
        
        b_cols = [f'Biomarker_Day{i}' for i in range(6)]
        df['Biomarker_Volatility'] = df[b_cols].std(axis=1)
        
       
        df['Sex_Male'] = 1 if str(df['Sex'].iloc[0]).lower() in ['male', 'm'] else 0
        
     
        scale_cols = ['Age', 'Baseline_Severity', 'Biomarker_Day0', 'Dose_Per_Kg']
        df[scale_cols] = scaler.transform(df[scale_cols])
        model_features = model.feature_names_in_
        X_final = df[model_features]
        
      
        prob = model.predict_proba(X_final)[0][1] # Probability of Class 1 (Responder)
        is_responder = prob > 0.5
        
        return {
            "prediction": "Responder" if is_responder else "Non-Responder",
            "probability_of_response": round(float(prob), 4),
            "dose_intensity": round(float(df['Dose_Per_Kg'].iloc[0]), 2), # Return this so doctor can see if dose is low
            "alert_clinician": not is_responder,
            "message": "Patient likely to respond." if is_responder else "ALERT: High risk of non-response. Check Dose/Kg."
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict_batch", response_model=List[BatchPredictionResponse])
async def predict_batch(file: UploadFile = File(...)):
    try:
        # 1. Read and validate CSV file
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
        contents = await file.read()
        try:
            df = pd.read_csv(io.BytesIO(contents))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid CSV file: {str(e)}")
        
        # 2. Normalize column names (convert to lowercase and replace spaces with underscores)
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        
        # 3. Validate required columns
        required_columns = {
            'age', 'sex', 'weight_kg', 'baseline_severity',
            'biomarker_day0', 'biomarker_day1', 'biomarker_day2',
            'biomarker_day3', 'biomarker_day4', 'biomarker_day5'
        }
        
        # Check for missing columns
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Missing required columns",
                    "missing_columns": list(missing_columns),
                    "available_columns": list(df.columns)
                }
            )
        
        results = []
        
        # 4. Process each row
        for idx, row in df.iterrows():
            try:
                # Create a PatientData instance for validation
                patient_data = {
                    'age': int(row['age']),
                    'sex': str(row['sex']).strip(),
                    'weight_kg': float(row['weight_kg']),
                    'baseline_severity': int(row['baseline_severity']),
                    'biomarker_day0': float(row['biomarker_day0']),
                    'biomarker_day1': float(row['biomarker_day1']),
                    'biomarker_day2': float(row['biomarker_day2']),
                    'biomarker_day3': float(row['biomarker_day3']),
                    'biomarker_day4': float(row['biomarker_day4']),
                    'biomarker_day5': float(row['biomarker_day5'])
                }
                
                # Get prediction using the existing predict_response function
                prediction_response = await predict_response(PatientData(**patient_data))
                prediction_dict = prediction_response.dict() if hasattr(prediction_response, 'dict') else prediction_response
                
                # Add patient identifier (use index if no ID column)
                patient_id = str(row.get('patient_id', f"patient_{idx + 1}"))
                
                # Create response dictionary
                result = {
                    "patient_id": patient_id,
                    "prediction": prediction_dict.get("prediction", "Error"),
                    "probability_of_response": prediction_dict.get("probability_of_response", 0.0),
                    "dose_intensity": prediction_dict.get("dose_intensity", 0.0),
                    "alert_clinician": prediction_dict.get("alert_clinician", True),
                    "message": prediction_dict.get("message", "Prediction completed")
                }
                results.append(result)
                
            except Exception as e:
                patient_id = str(row.get('patient_id', f"patient_{idx + 1}"))
                # Create error response that matches BatchPredictionResponse
                results.append({
                    "patient_id": patient_id,
                    "prediction": "Error",
                    "probability_of_response": 0.0,
                    "dose_intensity": 0.0,
                    "alert_clinician": True,
                    "message": f"Prediction failed: {str(e)}"
                })
        
        return results

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )