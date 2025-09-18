from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, computed_field
from typing import Annotated
import pandas as pd
import numpy as np
import joblib
import io

app = FastAPI(title="Revenue Predictor with EDA")

# ------------------- Load Model & Encoders -------------------
model = joblib.load("xgb_model.pkl")
le_adgroup = joblib.load("le_adgroup.pkl")
le_month = joblib.load("le_month.pkl")

# Required CSV columns (Conv_Rate added)
REQUIRED_COLUMNS = ['Impressions', 'Clicks', 'CTR', 'Conversions', 'Conv Rate', 
                    'Cost', 'CPC', 'Sale Amount', 'P&L', 'Ad Group', 'Month']


# ------------------- Input Schema with Computed Fields -------------------
class CampaignInput(BaseModel):
    Impressions: Annotated[int, Field(..., gt=0)]
    Clicks: Annotated[int, Field(..., ge=0)]
    CTR: Annotated[float, Field(..., ge=0)]
    Conversions: Annotated[int, Field(..., ge=0)]
    Conv_Rate: Annotated[float, Field(..., ge=0, description="Conversion rate = Conversions/Clicks")]
    Cost: Annotated[float, Field(..., ge=0)]
    CPC: Annotated[float, Field(..., ge=0)]
    Sale_Amount: Annotated[float, Field(..., ge=0)]
    PnL: Annotated[float, Field(...)]
    Ad_Group: Annotated[str, Field(...)]
    Month: Annotated[str, Field(...)]

    # -------- Computed Fields --------
    @computed_field
    @property
    def ROI(self) -> float:
        return round((self.Sale_Amount - self.Cost) / self.Cost if self.Cost > 0 else 0, 2)

    @computed_field
    @property
    def Profit_Margin(self) -> float:
        return round((self.Sale_Amount - self.Cost) / self.Sale_Amount if self.Sale_Amount > 0 else 0, 2)

    @computed_field
    @property
    def CPM(self) -> float:
        return round((self.Cost / self.Impressions) * 1000 if self.Impressions > 0 else 0, 2)

    @computed_field
    @property
    def Revenue_per_Click(self) -> float:
        return round(self.Sale_Amount / self.Clicks if self.Clicks > 0 else 0, 2)

    @computed_field
    @property
    def Revenue_per_Conversion(self) -> float:
        return round(self.Sale_Amount / self.Conversions if self.Conversions > 0 else 0, 2)
# ------------------- Root Endpoint -------------------
@app.get("/")
def read_root():
    return {"message": "👋 Welcome to the Multi-Touch Attribution (MTA) Revenue Predictor API!"}


# ------------------- CSV Validation -------------------
def validate_csv_columns(df: pd.DataFrame):
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise HTTPException(status_code=400, detail=f"Missing columns in CSV: {missing_cols}")


# ------------------- EDA Endpoint -------------------
@app.post("/eda")
async def run_eda(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        validate_csv_columns(df)

        # EDA Summary
        eda_summary = {
            "Shape": df.shape,
            "Missing_Values": df.isnull().sum().to_dict(),
            "Numeric_Summary": df.describe().to_dict(),
            "Top_Ad_Groups": df["Ad Group"].value_counts().head(5).to_dict(),
            "Top_Months": df["Month"].value_counts().head(5).to_dict(),
            "Correlation": df.corr(numeric_only=True).to_dict()
        }

        return JSONResponse(status_code=200, content=eda_summary)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


# ------------------- Predict from CSV -------------------
@app.post("/predict_from_csv")
async def predict_from_csv(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        validate_csv_columns(df)

        # Encode categorical features
        try:
            df["Ad Group Encoded"] = le_adgroup.transform(df["Ad Group"])
        except ValueError:
            df["Ad Group Encoded"] = -1

        try:
            df["Month Encoded"] = le_month.transform(df["Month"])
        except ValueError:
            df["Month Encoded"] = -1

        # Prepare features
        features = df.drop(columns=["Ad Group", "Month"])

        # Prediction
        df["Predicted_Revenue"] = model.predict(features)

        # Add KPIs
        df["ROI"] = (df["Predicted_Revenue"] - df["Cost"]) / df["Cost"]
        df["Profit_Margin"] = (df["Predicted_Revenue"] - df["Cost"]) / df["Predicted_Revenue"]
        df["CPM"] = df["Cost"] / (df["Impressions"] / 1000)
        df["Revenue_per_Click"] = df["Predicted_Revenue"] / df["Clicks"]
        df["Revenue_per_Conversion"] = df["Predicted_Revenue"] / df["Conversions"]

        return JSONResponse(status_code=200, content=df.to_dict(orient="records"))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


# ------------------- Predict from Manual Input -------------------
@app.post("/predict_from_manual")
def predict_from_manual(data: CampaignInput):
    try:
        # Encode categorical vars
        ad_group_encoded = le_adgroup.transform([data.Ad_Group])[0] if data.Ad_Group in le_adgroup.classes_ else -1
        month_encoded = le_month.transform([data.Month])[0] if data.Month in le_month.classes_ else -1

        # Prepare feature vector in correct order
        features = np.array([[
            data.Impressions, data.Clicks, data.CTR, data.Conversions, data.Conv_Rate,
            data.Cost, data.CPC, data.Sale_Amount, data.PnL,
            data.ROI, data.Profit_Margin, data.CPM,
            data.Revenue_per_Click, data.Revenue_per_Conversion,
            ad_group_encoded, month_encoded
        ]])

        # Predict revenue
        prediction = model.predict(features)[0]

        return {
            "Predicted_Revenue": round(float(prediction), 2),
            "Computed_KPIs": {
                "ROI": data.ROI,
                "Profit_Margin": data.Profit_Margin,
                "CPM": data.CPM,
                "Revenue_per_Click": data.Revenue_per_Click,
                "Revenue_per_Conversion": data.Revenue_per_Conversion
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing data: {str(e)}")
