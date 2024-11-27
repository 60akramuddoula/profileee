from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle
import json
import numpy as np

# Load the model
with open("banglore_home_prices_model.pickle", "rb") as f:
    model = pickle.load(f)

# Load the columns
with open("columns.json", "r") as f:
    data_columns = json.load(f)["data_columns"]

# Define FastAPI app
app = FastAPI()


# Define the input schema
class HousePriceInput(BaseModel):
    total_sqft: float
    bath: int
    bhk: int
    location: str


@app.get("/")
def root():
    return {"message": "Welcome to the Bangalore Home Price Prediction API"}


@app.post("/predict/")
def predict_home_price(input_data: HousePriceInput):
    try:
        # Prepare the input data for the model
        input_features = np.zeros(len(data_columns))

        # Fill the features with the input values
        input_features[0] = input_data.total_sqft
        input_features[1] = input_data.bath
        input_features[2] = input_data.bhk

        # One-hot encode the location if it exists in the data_columns
        if input_data.location in data_columns:
            location_index = data_columns.index(input_data.location)
            input_features[location_index] = 1

        # Predict the price using the model
        prediction = model.predict([input_features])
        result = prediction[0]

        return {"estimated_price": result}

    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error during prediction: {str(e)}"
        )
