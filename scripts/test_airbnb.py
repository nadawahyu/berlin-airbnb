import data_pipeline
import util as utils
import pandas as pd

def test_convert_price():
    # arrange
    config = utils.load_config()

    mock_data = {'Price':'6,000'}
    mock_data = pd.DataFrame(mock_data)

    expected_data = {'Price':6000}
    expected_data = pd.DataFrame(expected_data)

    # act
    processed_data = data_pipeline.convert_price(mock_data, config)

    # assert
    assert processed_data.equals(expected_data)