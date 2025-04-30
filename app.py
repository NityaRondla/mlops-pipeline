from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load model
model = joblib.load("model.pkl")

# Initialize FastAPI app
app = FastAPI()

# Define input data schema
class InputData(BaseModel):
    features: list

# Define predict endpoint
@app.post("/predict")
def predict(data: InputData):
    prediction = model.predict([data.features])
    return {"prediction": int(prediction[0])}
