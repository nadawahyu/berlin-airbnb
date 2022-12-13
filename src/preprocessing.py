import pandas as pd
import numpy as np
import util as util
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

def load_dataset(config_data: dict) -> pd.DataFrame:
    # Load every set of data
    x_train = util.pickle_load(config_data["train_set_path"][0])
    y_train = util.pickle_load(config_data["train_set_path"][1])

    x_valid = util.pickle_load(config_data["valid_set_path"][0])
    y_valid = util.pickle_load(config_data["valid_set_path"][1])

    x_test = util.pickle_load(config_data["test_set_path"][0])
    y_test = util.pickle_load(config_data["test_set_path"][1])
    # Concatenate x and y each set
    train_set = pd.concat([x_train, y_train],axis = 1)
    valid_set = pd.concat([x_valid, y_valid],axis = 1)
    test_set = pd.concat([x_test, y_test],axis = 1)

    # Return 3 set of data
    return train_set, valid_set, test_set

def nan_detector(set_data: pd.DataFrame) -> pd.DataFrame:
    # Create copy of set data
    set_data = set_data.copy()

    # Replace -1 with NaN
    set_data.replace(
        'nan', np.nan,
        inplace = True
    )

    # Return replaced set data
    return set_data

def remove_outliers(set_data):
    set_data = set_data.copy()
    list_of_set_data = list()

    for col_name in set_data.columns[:-1]:
        q1 = set_data[col_name].quantile(0.25)
        q3 = set_data[col_name].quantile(0.75)
        iqr = q3 - q1
        set_data_cleaned = set_data[~((set_data[col_name] < (q1 - 1.5 * iqr)) | (set_data[col_name] > (q3 + 1.5 * iqr)))].copy()
        list_of_set_data.append(set_data_cleaned.copy())
    
    set_data_cleaned = pd.concat(list_of_set_data)
    count_duplicated_index = set_data_cleaned.index.value_counts()
    used_index_data = count_duplicated_index[count_duplicated_index == (set_data.shape[1]-1)].index
    set_data_cleaned = set_data_cleaned.loc[used_index_data].drop_duplicates()

    return set_data_cleaned


def ImputerNumerical(set_data):
    set_data = set_data.copy().select_dtypes(['int', 'float'])

    # check whether imputer exists
    if imputer == None:
        # create imputer
        imputer = SimpleImputer(missing_values = np.nan,
                                strategy = 'median')
        
        # fit imputer
        imputer.fit(set_data)
        
    # transform data
    data_imputed = pd.DataFrame(imputer.transform(set_data))
    data_imputed.columns = set_data.columns
    data_imputed.index = set_data.index
    
    return data_imputed

def OHETransformer(set_data):
    set_data = set_data.copy().select_dtypes('str')

    # check encoder availability
    if encoder == None:
        #create encoder
        encoder = OneHotEncoder(#drop = 'if_binary',
                                handle_unknown = 'ignore')
        # fit
        encoder.fit(set_data)
        
        # extract ohe cols
        ohe_col = encoder.get_feature_names(set_data.columns)
        
    # transform data
    data_ohe = encoder.transform(set_data).toarray()
    data_ohe = pd.DataFrame(data_ohe,
                           columns = ohe_col,
                           index = set_data.index)
    
    return data_ohe

def concat_data(set_data):
    set_data = set_data.copy()
    set_data = pd.concat([ImputerNumerical(set_data), OHETransformer(set_data)])

    return set_data

if __name__ == "__main__":
    # 1. Load configuration file
    config_data = util.load_config()

    # 2. Load dataset
    train_set, valid_set, test_set = load_dataset(config_data)

    # 3. removing outliers
    train_set_cleaned = remove_outliers(train_set)

    # 5. impute data numerical
    train_set_cleaned_num = ImputerNumerical(train_set_cleaned)

    # 6. OHE for categorical data
    train_set_cleaned_cat = OHETransformer(train_set_cleaned)

    # 7. concat
    train_set_cleaned_concat = pd.concat([train_set_cleaned_num, train_set_cleaned_cat])


    # 19. Dumping dataset
    util.pickle_dump(
            train_set_cleaned_concat[config_data["predictors"]],
            config_data["train_feng_set_path"][0]
    )
    util.pickle_dump(
            train_set_cleaned_concat[config_data["label"]],
            config_data["train_feng_set_path"][1]
    )


    util.pickle_dump(
            valid_set[config_data["predictors"]],
            config_data["valid_feng_set_path"][0]
    )
    util.pickle_dump(
            valid_set[config_data["label"]],
            config_data["valid_feng_set_path"][1]
    )


    util.pickle_dump(
            test_set[config_data["predictors"]],
            config_data["test_feng_set_path"][0]
    )
    util.pickle_dump(
            test_set[config_data["label"]],
            config_data["test_feng_set_path"][1]
    )