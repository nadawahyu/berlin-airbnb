import streamlit as st
import requests
#import joblib
from PIL import Image
import numpy as np
import pandas as pd

# Add some information about the service
st.title("Airbnb Price Prediction in Berlin")
st.subheader("Just enter variables below and click Predict")

# Create form of input
with st.form(key = "air_data_form"):
    # Create select box input
    neighborhood = st.selectbox(
        label = "Pick preferred neighborhood",
        options = (
            'Steglitz - Zehlendorf', 'Pankow', 'Friedrichshain-Kreuzberg',
            'Mitte', 'Lichtenberg', 'Neukoelln', 'Tempelhof - Schoeneberg',
            'Reinickendorf', 'Charlottenburg-Wilm.', 'Treptow - Koepenick',
            'Marzahn - Hellersdorf', 'Spandau'
            )
        )

    room_type = st.selectbox(
        label = "Pick preferred room type",
        options = (
            'Private room', 
            'Entire home/apt', 
            'Shared room'
            )
        )

    instant_bookable = st.selectbox(
        label = "Pick instant bookable preference",
        options = (
            't', 
            'f'
            )
        )

    # Create box for number input
    accomodates = st.number_input(
        label = "Enter num of people can be accomodated:",
        min_value = 1,
        max_value = 16,
        help = "Value ranges from 1 to 16"
    )

    latitude = st.number_input(
        label = "Enter latitude:",
        min_value = 52,
        max_value = 53,
        help = "Value ranges from 52 to 53"
    )

    longitude = st.number_input(
        label = "Enter longitude:",
        min_value = 13,
        max_value = 14,
        help = "Value ranges from 13 to 14"
    )

    guests = st.number_input(
        label = "Enter max guest:",
        min_value = 1,
        max_value = 16,
        help = "Value ranges from 0 to 1 16"
    )
    
    sqft = st.number_input(
        label = "Enter area [sqft]:",
        min_value = 300,
        max_value = 1250,
        help = "Value range from 300 to 1250"
    )

    min_nights = st.number_input(
        label = "Enter min nights:",
        min_value = 0,
        max_value = 1000,
        help = "Value range from 0 to 1000"
    )

    bedrooms = st.number_input(
        label = "Enter num of bedrooms:",
        min_value = 0,
        max_value = 10,
        help = "Value range from 0 to 10"
    )

    beds = st.number_input(
        label = "Enter num of beds:",
        min_value = 0,
        max_value = 22,
        help = "Value range from 400 to 60000"
    )

    bathrooms = st.number_input(
        label = "Enter num of bathrooms:",
        min_value = 0.0,
        max_value = 8.5,
        step = 0.5,
        help = "Value range from 0 to 8.5"
    )

    rating_overall = st.number_input(
        label = "Enter overall rating:",
        min_value = 1,
        max_value = 10,
        help = "Value range from 2 to 10"
    )

    rating_accuracy = st.number_input(
        label = "Enter Raw Ethanol Value:",
        min_value = 1,
        max_value = 10,
        help = "Value range from 2 to 10"
    )

    rating_cleanliness = st.number_input(
        label = "Enter rating for cleanliness:",
        min_value = 1,
        max_value = 10,
        help = "Value range from 2 to 10"
    )

    rating_checkin = st.number_input(
        label = "Enter rating for ease for checkin:",
        min_value = 1,
        max_value = 10,
        help = "Value range from 2 to 10"
    )

    rating_communication = st.number_input(
        label = "Enter rating for communication:",
        min_value = 1,
        max_value = 10,
        help = "Value range from 2 to 10"
    )

    rating_location = st.number_input(
        label = "Enter rating for location:",
        min_value = 1,
        max_value = 10,
        help = "Value range from 2 to 10"
    )

    rating_value = st.number_input(
        label = "Enter rating for value:",
        min_value = 1,
        max_value = 10,
        help = "Value range from 2 to 10"
    )
    
    # Create button to submit the form
    submitted = st.form_submit_button("Predict")

    # Condition when form submitted
    if submitted:
        # Create dict of all data in the form
        raw_data = {
                'accomodates': accomodates,
                'guests': guests,
                'neighborhood': neighborhood,
                'room_type': room_type,
                'instant_bookable': instant_bookable,
                'latitude': latitude,
                'longitude': longitude,
                'sqft': sqft,
                'min_nights': min_nights,
                'bedrooms': bedrooms,
                'beds': beds,
                'bathrooms': bathrooms,
                'rating_overall': rating_overall,
                'rating_accuracy': rating_accuracy,
                'rating_cleanliness': rating_cleanliness,
                'rating_checkin': rating_checkin,
                'rating_communication': rating_communication,
                'rating_location': rating_location,
                'rating_value': rating_value
        }

        # Create loading animation while predicting
        with st.spinner("Sending data to prediction server ..."):
            res = requests.post("http://api_backend:8501/predict", json = raw_data).json()
            
        # Parse the prediction result
        if res["error_msg"] != "":
            st.error("Error Occurs While Predicting: {}".format(res["error_msg"]))
        else:
            if res["res"] != "Tidak ada api.":
                st.warning("Ada api.")
            else:
                st.success("Tidak ada api.")