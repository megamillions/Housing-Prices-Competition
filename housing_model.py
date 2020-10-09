import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

'''
Validation MAE for Random Forest Model: 15,601
'''

# True - testing against train set to measure accuracy.
# False - testing against test set and save for submission.
is_training = True

# Path of the file to read.
iowa_file_path = 'train.csv'
home_data = pd.read_csv(iowa_file_path)

# Create target object and call it y
y = home_data.SalePrice

# Features with numerical missing data.
num_na_features = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt',
                   'BsmtFinSF1', 'TotalBsmtSF', 'GarageCars']

# Ordinal features to be mapped to 5-point score.
ordinal_5_features = ['ExterCond', 'HeatingQC', 'KitchenQual']

# Ordinal features to be mapped to 3-point score.
ordinal_3_features = ['PavedDrive']

# Ordinal features to be mapped to its respective score, using null as 0.
ord_5a_na_features = ['BsmtExposure']

ord_5b_na_features = ['PoolQC']

ord_6_na_features = ['BsmtQual', 'BsmtCond', 'FireplaceQu',
                     'GarageQual', 'GarageCond']

ord_7_na_features = ['BsmtFinType1', 'BsmtFinType2']

# Features with categorical data.
categorical_features = ['MSSubClass', 'MSZoning', 'Street',
                     'LotShape', 'LandContour', 'Utilities',
                     'LotConfig', 'LandSlope', 'Neighborhood',
                     'Condition1', 'Condition2', 'BldgType',
                     'HouseStyle', 'RoofStyle', 'RoofMatl',
                     'Exterior1st', 'Exterior2nd', 'Foundation',
                     'Heating', 'CentralAir', 'PavedDrive',
                     'SaleType', 'SaleCondition']

# Features with categorical missing data to be assigned dummy variables.
cat_na_features = ['Alley', 'MasVnrType', 'Electrical',
                   'GarageType', 'GarageFinish', 'Fence',
                   'MiscFeature']

# Features with no need to edit.
features = ['OverallQual', 'OverallCond', 'YearBuilt',
            '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
            'GrLivArea', 'FullBath', 'HalfBath',
            'BedroomAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
            'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',
            '3SsnPorch', 'PoolArea', 'MiscVal']

def map_na_5a(dataset, features):
    
    for feature in features:
        
        rankings = {
            'Gd' : 4,
            'Av' : 3,
            'Mn' : 2,
            'No' : 1,
            np.nan : 0}
        
        dataset[feature] = dataset[feature].map(rankings)
        
    return features

def map_na_5b(dataset, features):
    
    for feature in features:
        
        rankings = {
            'Ex' : 4,
            'Gd' : 3,
            'TA' : 2,
            'Fa' : 1,
            np.nan : 0}
    
        dataset[feature] = dataset[feature].map(rankings)
        
    return features

def map_na_6(dataset, features):
    
    for feature in features:

        rankings = {
            'Ex' : 5,
            'Gd' : 4,
            'TA' : 3,
            'Fa' : 2,
            'Po' : 1,
            np.nan : 0}    
        
        dataset[feature] = dataset[feature].map(rankings)
    
    return features

def map_na_7(dataset, features):

    for feature in features:
    
        rankings = {
            'GLQ' : 6,
            'ALQ' : 5,
            'BLQ' : 4,
            'Rec' : 3,
            'LwQ' : 2,
            'Unf' : 1,
            np.nan : 0}
        
        dataset[feature] = dataset[feature].map(rankings)
    
    return features

# Numericize rankings into a 5-point score.
def rank_ordinal_5(series):
    
    rankings = {
        'Ex' : 4,
        'Gd' : 3,
        'TA' : 2,
        'Fa' : 1,
        'Po' : 0}
    
    return series.map(rankings)

# Numericize rankings into a 3-point score.
def rank_ordinal_3(series):
    
    rankings = {
        'Y' : 2,
        'P' : 1,
        'N' : 0}
    
    return series.map(rankings)

# Returns the X used in testing and training.
def clean_data(dataset):
    
    # Preprocessing and imputation for missing numerical data.
    mean_imputer = SimpleImputer()

    for feature in num_na_features:
        mean_imputer = mean_imputer.fit(dataset[[feature]])
        dataset[feature] = mean_imputer.transform(dataset[[feature]]).ravel()

    # Apply data cleaning to ordinal features.
    for feature in ordinal_5_features:
        
        dataset[feature] = rank_ordinal_5(dataset[feature])

        # Impute any missing values.
        mean_imputer = mean_imputer.fit(dataset[[feature]])
        dataset[feature] = mean_imputer.transform(dataset[[feature]]).ravel()
    
    for feature in ordinal_3_features:
        dataset[feature] = rank_ordinal_3(dataset[feature])

    # Clean ordinal features by assigning numericals including to null values.
    map_na_5a(dataset, ord_5a_na_features)
    map_na_5b(dataset, ord_5b_na_features)
    map_na_6(dataset, ord_6_na_features)
    map_na_7(dataset, ord_7_na_features)

    # Apply data cleaning to categorical features.
    oh_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    oh_cols = pd.DataFrame(oh_encoder.fit_transform(dataset[categorical_features]))

    oh_cols.index = dataset.index
    
    # Create dummies for features with nan.
    hd_dummies = pd.get_dummies(
        dataset[cat_na_features], columns=cat_na_features)

    # Combine all features together.
    X = pd.concat([dataset[features], dataset[num_na_features],
               dataset[ordinal_5_features], dataset[ordinal_3_features],
               dataset[ord_7_na_features], dataset[ord_6_na_features],
               dataset[ord_5a_na_features], dataset[ord_5b_na_features],
               oh_cols, hd_dummies],
              axis=1)

    return X

# Combine all features together.
X = clean_data(home_data)

if is_training:
    
    # Split into validation and training data.
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

    # Define the model. Set random_state to 1.
    model = RandomForestRegressor(random_state=1)
    model.fit(train_X, train_y)
    
    val_predictions = model.predict(val_X)
    val_mae = mean_absolute_error(val_predictions, val_y)
    print("Validation MAE for Random Forest Model: {:,.0f}".
          format(val_mae))

else:
    
    # Path to file you will use for predictions.
    test_data_path = 'test.csv'
    
    # Read test data file using pandas.
    test_data = pd.read_csv(test_data_path)
    
    # Create test_X including only the columns you used for prediction.
    test_X = clean_data(test_data)

    # Check for any columns to drop that may not appear in the test set.
    for column in X.columns:
        
        matching = False
        
        for test_column in test_X.columns:
            if column == test_column:
                matching = True
                
        if matching == False:
            del X[column]
    
    # Create a new Random Forest model.
    model_full = RandomForestRegressor(random_state=1)
    
    # Fit rf_model_full on all data from the training data.
    model_full.fit(X, y)
        
    # Make predictions which we will submit. 
    test_preds = model_full.predict(test_X)
    
    # Save predictions in format used for competition.
    output = pd.DataFrame({'Id': test_data.Id,
                           'SalePrice': test_preds})
    
    output.to_csv('submission.csv', index=False)
    print('File successfully saved!')