from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
import os
import copy
import util as util

def read_raw_data(config: dict) -> pd.DataFrame:
    # Create variable to store raw dataset
    raw_dataset = pd.DataFrame()

    # Raw dataset dir
    raw_dataset_dir = config['raw_dataset_dir']

    # Look and load add CSV files
    for i in tqdm(os.listdir(raw_dataset_dir)):
        raw_dataset = pd.concat([pd.read_csv(raw_dataset_dir + i), raw_dataset])
    
    # Return raw dataset
    return raw_dataset

def check_data(input_data, params, api = False):
    input_data = copy.deepcopy(input_data)
    params = copy.deepcopy(params)

    if not api:
        # rename columns
        input_data.rename({
            'Accomodates': 'accomodates',
            'Guests Included': 'guests',
            'Neighborhood Group': 'neighborhood',
            'Room Type': 'room_type',
            'Instant Bookable': 'instant_bookable',
            'Latitude': 'latitude',
            'Longitude': 'longitude',
            'SSquare Feet': 'sqft',
            'Price': 'price',
            'Min Nights': 'min_nights',
            'Bedrooms': 'bedrooms',
            'Beds': 'beds',
            'Bathrooms': 'bathrooms',
            'Overall Rating': 'rating_overall',
            'Accuracy Rating': 'rating_accuracy',
            'Cleanliness Rating': 'rating_cleanliness',
            'Checkin Rating': 'rating_checkin',
            'Communication Rating': 'rating_communication',
            'Location Rating': 'rating_location',
            'Value Rating': 'rating_value',
        }, axis=1, inplace=True)

        # Check data types
        assert input_data.select_dtypes('datetime').columns.to_list() == \
            params['datetime_columns'], 'an error occurs in datetime column(s).'
        assert input_data.select_dtypes('object').columns.to_list() == \
            params['object_columns'], 'an error occurs in object column(s).'
        assert input_data.select_dtypes('int').columns.to_list() == \
            params['int64_columns'], 'an error occurs in int64 column(s).'
        assert input_data.select_dtypes('float').columns.to_list() == \
            params['float_columns'], 'an error occurs in float column(s).'
    else:
        # In case checking data from api
        # Predictor that has object dtype only 
        object_columns = params['object_columns']
        del object_columns[1:]

        # Max column not used as predictor
        int_columns = params['int64_columns']
        del int_columns[-1]

        float_columns = params['float_columns']
        del int_columns[-1]

        # Check data types
        assert input_data.select_dtypes('object').columns.to_list() == \
            object_columns, 'an error occurs in object column(s).'
        assert input_data.select_dtypes('int').columns.to_list() == \
            int_columns, 'an error occurs in int64 column(s).'
        assert input_data.select_dtypes('float').columns.to_list() == \
            float_columns, 'an error occurs in float column(s).'

    assert input_data['latitude'].between(params['range_latitude'][0], params['range_latitude'][1]).sum() == \
        len(input_data), 'an error occurs in latitude range.'
    assert input_data['longitude'].between(params['range_longitude'][0], params['range_longitude'][1]).sum() == \
        len(input_data), 'an error occurs in longitude range.'
    assert input_data['accomodates'].between(params['range_accomodates'][0], params['range_accomodates'][1]).sum() == \
        len(input_data), 'an error occurs in accomodates range.'
    assert input_data['rooms'].between(params['range_bathrooms'][0], params['range_bathrooms'][1]).sum() == \
        len(input_data), 'an error occurs in rooms range.'
    assert input_data['beds'].between(params['range_beds'][0], params['range_beds'][1]).sum() == \
        len(input_data), 'an error occurs in beds range.'
    assert input_data['sqft'].between(params['range_sqft'][0], params['range_sqft'][1]).sum() == \
        len(input_data), 'an error occurs in square feet range.'
    assert input_data['guests'].between(params['range_guests'][0], params['range_guests'][1]).sum() == \
        len(input_data), 'an error occurs in guests range.'
    assert input_data['min_nights'].between(params['range_nights'][0], params['range_nights'][1]).sum() == \
        len(input_data), 'an error occurs in nights range.'
    assert input_data['rating_overall'].between(params['range_overall'][0], params['range_overall'][1]).sum() == \
        len(input_data), 'an error occurs in overall rating range.'
    assert input_data['rating_accuracy'].between(params['range_accuracy'][0], params['range_accuracy'][1]).sum() == \
        len(input_data), 'an error occurs in accuracy rating range.'
    assert input_data['rating_cleanliness'].between(params['range_cleanliness'][0], params['range_cleanliness'][1]).sum() == \
        len(input_data), 'an error occurs in cleanliness rating range.'
    assert input_data['rating_checkin'].between(params['range_checkin'][0], params['range_checkin'][1]).sum() == \
        len(input_data), 'an error occurs in checkin rating range.'
    assert input_data['rating_communication'].between(params['range_comm'][0], params['range_comm'][1]).sum() == \
        len(input_data), 'an error occurs in communication rating range.'
    assert input_data['rating_location'].between(params['range_loc'][0], params['range_loc'][1]).sum() == \
        len(input_data), 'an error occurs in location rating range.'
    assert input_data['rating_value'].between(params['range_value'][0], params['range_value'][1]).sum() == \
        len(input_data), 'an error occurs in value rating range.'

if __name__ == '__main__':
    # 1. Load configuration file
    config_data = util.load_config()

    # 2. Read all raw dataset
    raw_dataset = read_raw_data(config_data)

    # 3. Reset index
    raw_dataset.reset_index(
        inplace = True,
        drop = True
    )

    # 4. Save raw dataset
    util.pickle_dump(
        raw_dataset,
        config_data['raw_dataset_path']
    )
    
    # 5. Handling variable price
    raw_dataset['Price'] = raw_dataset['Price'].replace({',':''}, regex=True).apply(pd.to_numeric,1)

    # 6. Handling overall rating
    raw_dataset['Overall Rating'] = raw_dataset['Overall Rating']/10

    # 7. Handling neighborhood group
    raw_dataset['Neighborhood Group'] = raw_dataset['Neighborhood Group'].replace({'NeukÃ¶lln':'Neukoelln', 
                                    'Tempelhof - SchÃ¶neberg':'Tempelhof - Schoeneberg',
                                    'Treptow - KÃ¶penick':'Treptow - Koepenick'})
    # 8. Handling non-germany data
    raw_dataset = raw_dataset[raw_dataset['Country'] == 'Germany']

    # 12. Check data definition
    check_data(raw_dataset, config_data)

    # 13. Splitting input output
    x = raw_dataset[config_data['predictors']].copy()
    y = raw_dataset['Price'].copy()

    # 14. Splitting train test
    x_train, x_test, \
    y_train, y_test = train_test_split(
        x, y,
        test_size = 0.3,
        random_state = 42,
        stratify = y
    )

    # 15. Splitting test valid
    x_valid, x_test, \
    y_valid, y_test = train_test_split(
        x_test, y_test,
        test_size = 0.5,
        random_state = 42,
        stratify = y_test
    )

    # 16. Save train, valid and test set
    util.pickle_dump(x_train, config_data['train_set_path'][0])
    util.pickle_dump(y_train, config_data['train_set_path'][1])

    util.pickle_dump(x_valid, config_data['valid_set_path'][0])
    util.pickle_dump(y_valid, config_data['valid_set_path'][1])

    util.pickle_dump(x_test, config_data['test_set_path'][0])
    util.pickle_dump(y_test, config_data['test_set_path'][1])