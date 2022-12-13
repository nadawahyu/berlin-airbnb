import preprocessing
import util as utils
import pandas as pd
import numpy as np

def test_nan_detector():
    # Arrange
    mock_data = {"pm10" : [23, -1, 50, 53, -1, 20]}
    mock_data = pd.DataFrame(mock_data)
    expected_data = {"pm10" : [23, np.nan, 50, 53, np.nan, 20]}
    expected_data = pd.DataFrame(expected_data)

    # Act
    processed_data = preprocessing.nan_detector(mock_data)

    # Assert
    assert processed_data.equals(expected_data)

def test_ohe_transform():
    # Arrange
    config = utils.load_config()
    ohe_object = utils.pickle_load(config["ohe_stasiun_path"])
    mock_data = {
        "stasiun" : [
            "DKI1 (Bunderan HI)", "DKI2 (Kelapa Gading)", 
            "DKI3 (Jagakarsa)", "DKI4 (Lubang Buaya)", 
            "DKI5 (Kebon Jeruk) Jakarta Barat"]}
    mock_data = pd.DataFrame(mock_data)
    expected_data = {
        "DKI1 (Bunderan HI)" : [1, 0, 0, 0, 0], "DKI2 (Kelapa Gading)" : [0, 1, 0, 0, 0], "DKI3 (Jagakarsa)" : [0, 0, 1, 0, 0], "DKI4 (Lubang Buaya)" : [0, 0, 0, 1, 0], "DKI5 (Kebon Jeruk) Jakarta Barat" : [0, 0, 0, 0, 1]}
    expected_data = pd.DataFrame(expected_data)
    expected_data = expected_data.astype(float)

    # Act
    processed_data = preprocessing.ohe_transform(mock_data, "stasiun", ohe_object)

    # Assert
    assert processed_data.equals(expected_data)
