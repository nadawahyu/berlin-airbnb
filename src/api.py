from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import pandas as pd
import util as util
import numpy as np
import data_pipeline as data_pipeline
import preprocessing as preprocessing

config_data = util.load_config()
ohe_stasiun = util.pickle_load(config_data["ohe_stasiun_path"])
model_data = util.pickle_load(config_data["production_model_path"])

class api_data(BaseModel):
    accomodates: int
    guests: int
    neighborhood: str
    room_type: str
    instant_bookable: str
    latitude: float
    longitude: float
    sqft: float
    price: float
    min_nights: float
    bedrooms: float
    beds: float
    bathrooms: float
    rating_overall: float
    rating_accuracy: float
    rating_cleanliness: float
    rating_checkin: float
    rating_communication: float
    rating_location: float
    rating_value: float

app = FastAPI()

@app.get("/")
def home():
    return "Hello, FastAPI up!"

@app.post("/predict/")
def predict(data: api_data):    
    # Convert data api to dataframe
    data = pd.DataFrame(data).set_index(0).T.reset_index(drop = True)

    # Convert dtype
    data = pd.concat(
        [
            data[config_data["predictors"][0]],
            data[config_data["predictors"][1:]].astype(int)
        ],
        axis = 1
    )

    # Check range data
    try:
        data_pipeline.check_data(data, config_data, True)
    except AssertionError as ae:
        return {"res": [], "error_msg": str(ae)}
    
    # Predict data
    y_pred = model_data["model_data"]["model_object"].predict(data)

    # Inverse tranform
    y_pred = list(le_encoder.inverse_transform(y_pred))[0] 

    return {"res" : y_pred, "error_msg": ""}

if __name__ == "__main__":
    uvicorn.run("api:app", host = "0.0.0.0", port = 8080)