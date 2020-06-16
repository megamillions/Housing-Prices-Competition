import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

'''

Validation MAE when not specifying max_leaf_nodes: 24,059
Validation MAE for best value of max_leaf_nodes: 22,859
Validation MAE for Random Forest Model: 16,001

'''

# True - testing against train set to measure accuracy.
# False - testing against test set and save for submission.
is_accuracy = True

# Path of the file to read.
iowa_file_path = 'train.csv'
home_data = pd.read_csv(iowa_file_path)

# Average out any na values.
def average_na(series):
    return series.fillna(series.mean())

# Numericize rankings into a 7-point score, assigning null to 0.
def ordinal_7_na(series):

    rankings = {
        'GLQ' : 6,
        'ALQ' : 5,
        'BLQ' : 4,
        'Rec' : 3,
        'LwQ' : 2,
        'Unf' : 1,
        np.nan : 0}
    
    return series.map(rankings)

# Numericize rankings into a 6-point score, assigning null to 0.
def ordinal_6_na(series):
    
    rankings = {
        'Ex' : 5,
        'Gd' : 4,
        'TA' : 3,
        'Fa' : 2,
        'Po' : 1,
        np.nan : 0}
    
    return series.map(rankings)

# Numericize rankings into a 5-point score, assigning null to 0.
def ordinal_5a_na(series):
    
    rankings = {
        'Gd' : 4,
        'Av' : 3,
        'Mn' : 2,
        'No' : 1,
        np.nan : 0}
    
    return series.map(rankings)

def ordinal_5b_na(series):
    
    rankings = {
        'Ex' : 4,
        'Gd' : 3,
        'TA' : 2,
        'Fa' : 1,
        np.nan : 0}
    
    return series.map(rankings)

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

# Returns specified model.
def get_model(X, y, m):
    
    # Split into validation and training data.
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

    # Specify and fit model.
    iowa_model = DecisionTreeRegressor(random_state=1)
    iowa_model.fit(train_X, train_y)
    
    # Make validation predictions and calculate mean absolute error.
    val_predictions = iowa_model.predict(val_X)
    val_mae = mean_absolute_error(val_predictions, val_y)
    print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".
          format(val_mae))
    
    # Using best value for max_leaf_nodes.
    iowa_max_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
    iowa_max_model.fit(train_X, train_y)
    
    val_max_predictions = iowa_max_model.predict(val_X)
    val_max_mae = mean_absolute_error(val_max_predictions, val_y)
    print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".
          format(val_max_mae))
    
    # Define the model. Set random_state to 1.
    rf_model = RandomForestRegressor(random_state=1)
    rf_model.fit(train_X, train_y)
    
    rf_val_predictions = rf_model.predict(val_X)
    rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)
    print("Validation MAE for Random Forest Model: {:,.0f}".
          format(rf_val_mae))

    # Returns specified model's dataframe.
    results = [val_predictions, val_max_predictions, rf_val_predictions]
    return results[m]

# Create target object and call it y
y = home_data.SalePrice

'''
# FEATURES NOT IMPLEMENTED

# Features with binary categorical data. - 1-hot encode.
bicat_features = ['Street', 'CentralAir']

# Features with categorical data. - 1-hot encode.
cat_features = ['SaleCondition']

# Features with categorical missing data. - 1-hot encode.
cat_na_features = ['Alley', 'MasVnrType', 'Electrical',
                   'GarageType', 'GarageFinish', 'Fence',
                   'MiscFeature']

'''

# Features with numerical missing data.
num_na_features = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt',
                   'BsmtFinSF1', 'TotalBsmtSF', 'GarageCars']

# Ordinal features to be mapped to 5-point score.
ordinal_5_features = ['ExterQual', 'ExterCond', 'HeatingQC',
                      'KitchenQual']

# Ordinal features to be mapped to 3-point score.
ordinal_3_features = ['PavedDrive']

# Ordinal features to be mapped to 7-point score, including null.
ord_7_na_features = ['BsmtFinType1', 'BsmtFinType2']

# Ordinal features to be mapped to 6-point score, including null.
ord_6_na_features = ['BsmtQual', 'BsmtCond', 'FireplaceQu',
                     'GarageQual', 'GarageCond']

# Ordinal features to be mapped to 5-point score, including null.
ord_5a_na_features = ['BsmtExposure']
ord_5b_na_features = ['PoolQC']

# Features with no need to edit.
features = ['LotArea', 'OverallQual', 'OverallCond',
            'YearBuilt', '1stFlrSF', '2ndFlrSF',
            'LowQualFinSF', 'GrLivArea', 'FullBath',
            'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
            'TotRmsAbvGrd', 'Fireplaces', 'WoodDeckSF',
            'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
            'ScreenPorch', 'PoolArea', 'MiscVal']

# Apply data cleaning to numerical features.
for feature in num_na_features:
   home_data[feature] = average_na(home_data[feature])

# Apply data cleaning to ordinal features.
for feature in ordinal_5_features:
    home_data[feature] = rank_ordinal_5(home_data[feature])

for feature in ordinal_3_features:
    home_data[feature] = rank_ordinal_3(home_data[feature])

# Apply data cleaning to ordinal features with missing values.
for feature in ord_7_na_features:
    home_data[feature] = ordinal_7_na(home_data[feature])
    
for feature in ord_6_na_features:
    home_data[feature] = ordinal_6_na(home_data[feature])

for feature in ord_5a_na_features:
    home_data[feature] = ordinal_5a_na(home_data[feature])
    
for feature in ord_5b_na_features:
    home_data[feature] = ordinal_5b_na(home_data[feature])

# Combine all features together.
X = pd.concat([home_data[features], home_data[num_na_features],
               home_data[ordinal_5_features], home_data[ordinal_3_features],
               home_data[ord_7_na_features], home_data[ord_6_na_features],
               home_data[ord_5a_na_features], home_data[ord_5b_na_features]],
              axis=1)

if is_accuracy:
    
    # Returns the RFR model.
    rf = get_model(X, y, 2)
    
else:
        
    # Create a new Random Forest model.
    rf_model_on_full_data = RandomForestRegressor()
    
    # Fit rf_model_on_full_data on all data from the training data.
    rf_model_on_full_data.fit(X, y)
    
    # Path to file you will use for predictions.
    test_data_path = 'test.csv'
    
    # Read test data file using pandas.
    test_data = pd.read_csv(test_data_path)
    
    # Apply data cleaning.    
    for feature in ordinal_5_features:
        test_data[feature] = rank_ordinal_5(test_data[feature])
    
    for feature in ordinal_3_features:        
        test_data[feature] = rank_ordinal_5(test_data[feature])
    
    # Create test_X including only the columns you used for prediction.
    test_X = pd.concat(
        [test_data[features], test_data[num_na_features],
         test_data[ordinal_5_features], test_data[ordinal_3_features]],
        axis=1)
    
    # Make predictions which we will submit. 
    test_preds = rf_model_on_full_data.predict(test_X)
    
    # Save predictions in format used for competition.
    output = pd.DataFrame({'Id': test_data.Id,
                           'SalePrice': test_preds})
    
    output.to_csv('submission.csv', index=False)
    print('File successfully saved!')