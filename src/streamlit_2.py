import streamlit as st
import requests
import joblib
from PIL import Image
import numpy as np
import pandas as pd

# Add some information about the service
st.title("Airbnb Price Prediction in Berlin")
st.subheader("Just enter variables below and click Predict")