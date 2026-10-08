import sys
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.pipeline.predict_pipeline import CustomData , PredictPipeline
from src.logger import logging

app = FastAPI(
    title = "Churn prediction API",
    description = "FastAPI endpoint for Customer Churn prediction",
    version = "1.0.0"
)

class CustomerDataSchema(BaseModel):
    gender: str = Field(..., example="Female")
    SeniorCitizen: int = Field(..., example=0, description="0 for No, 1 for Yes")
    Partner: str = Field(..., example="Yes")
    Dependents: str = Field(..., example="No")
    tenure: int = Field(..., ge=0, example=12, description="Number of months")
    PhoneService: str = Field(..., example="Yes")
    MultipleLines: str = Field(..., example="No phone service")
    InternetService: str = Field(..., example="DSL")
    OnlineSecurity: str = Field(..., example="No")
    OnlineBackup: str = Field(..., example="Yes")
    DeviceProtection: str = Field(..., example="No")
    TechSupport: str = Field(..., example="No")
    StreamingTV: str = Field(..., example="No")
    StreamingMovies: str = Field(..., example="No")
    Contract: str = Field(..., example="Month-to-month")
    PaperlessBilling: str = Field(..., example="Yes")
    PaymentMethod: str = Field(..., example="Electronic check")
    MonthlyCharges: float = Field(..., ge=0.0, example=29.85)
    TotalCharges: float = Field(..., ge=0.0, example=29.85)

@app.get("/")
def home():
    return {'health_check':'OK'}

@app.post("/predict")
def predict_datapoint(data:CustomerDataSchema):
    try:
        custom_data = CustomData(
            gender=data.gender,
            SeniorCitizen=data.SeniorCitizen,
            Partner=data.Partner,
            Dependents=data.Dependents,
            tenure=data.tenure,
            PhoneService=data.PhoneService,
            MultipleLines=data.MultipleLines,
            InternetService=data.InternetService,
            OnlineSecurity=data.OnlineSecurity,
            OnlineBackup=data.OnlineBackup,
            DeviceProtection=data.DeviceProtection,
            TechSupport=data.TechSupport,
            StreamingTV=data.StreamingTV,
            StreamingMovies=data.StreamingMovies,
            Contract=data.Contract,
            PaperlessBilling=data.PaperlessBilling,
            PaymentMethod=data.PaymentMethod,
            MonthlyCharges=data.MonthlyCharges,
            TotalCharges=data.TotalCharges
        )

        pred_df = custom_data.get_data_as_df()

        predict_pipeline = PredictPipeline()

        result = predict_pipeline.predict(pred_df)

        prediction_val = int(result[0])

        logging.info("predict successfully")
        return {
            "prediction" : prediction_val,
            "label" :"Churn" if prediction_val==1 else "Stay"
        }

    except Exception as e:
        logging.error(f"Error occurred during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
