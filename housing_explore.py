import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

'''

3rd model score: 16685.82146
Validation MAE when not specifying max_leaf_nodes: 23,874
Validation MAE for best value of max_leaf_nodes: 22,699
Validation MAE for Random Forest Model: 15,877
Validation MAE for RFM with SimpleImputer: 16,175

'''

# True - testing against train set to measure accuracy.
# False - testing against test set and save for submission.
is_accuracy = True

# Path of the file to read.
iowa_file_path = 'train.csv'
home_data = pd.read_csv(iowa_file_path)

# Alternate method to average out any na values. Not in use.
def average_na(series):
    return series.fillna(series.mean())

# Numericize rankings into a 5-point score.
def score_5_ranking(series):
    
    rankings = {
        'Ex' : 4,
        'Gd' : 3,
        'TA' : 2,
        'Fa' : 1,
        'Po' : 0}
    
    return series.map(rankings)

# Numericize rankings into a 3-point score.
def score_3_ranking(series):
    
    rankings = {
        'Y' : 2,
        'P' : 1,
        'N' : 0}
    
    return series.map(rankings)

# Returns specified model.
def get_model(X, y, m):
    
    # Split into validation and training data.
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
    
    # Apply data cleaning to numerical missing values.
    imputed_train_X, imputed_val_X = impute_X(train_X, val_X)
    
    # Imputation removes column names.
    imputed_train_X.columns = train_X.columns
    imputed_val_X.columns = val_X.columns
    
    # Specify model and fit model.
    iowa_model = DecisionTreeRegressor(random_state=1)
    iowa_model.fit(imputed_train_X, train_y)
    
    # Make validation predictions and calculate mean absolute error.
    val_predictions = iowa_model.predict(imputed_val_X)
    val_mae = mean_absolute_error(val_predictions, val_y)
    print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".
          format(val_mae))
    
    # Using best value for max_leaf_nodes.
    iowa_max_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
    iowa_max_model.fit(imputed_train_X, train_y)
    
    val_max_predictions = iowa_max_model.predict(imputed_val_X)
    val_max_mae = mean_absolute_error(val_max_predictions, val_y)
    print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".
          format(val_max_mae))
    
    # Define the model. Set random_state to 1.
    rf_model = RandomForestRegressor(random_state=1)
    rf_model.fit(imputed_train_X, train_y)
    
    rf_val_predictions = rf_model.predict(imputed_val_X)
    rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)
    print("Validation MAE for Random Forest Model: {:,.0f}".
          format(rf_val_mae))

    # Returns specified model's dataframe.
    results = [val_predictions, val_max_predictions, rf_val_predictions]
    return results[m]

# Applies a SimpleImputer to the train and validation X data sets.
def impute_X(train_X_data, val_X_data):

    # SimpleImputer averages out the missing data.
    imputer = SimpleImputer()

    fit_transform = pd.DataFrame(imputer.fit_transform(train_X_data))
    transform = pd.DataFrame(imputer.transform(val_X_data))
    
    return fit_transform, transform

# Create target object and call it y
y = home_data.SalePrice

'''
# FEATURES NOT IMPLEMENTED BELOW

# Features with binary categorical data.
bicat_features = ['Street', 'CentralAir']

# Features with categorical data.
cat_features = ['SaleCondition']

# Features with categorical missing data.
cat_na_features = ['Alley', 'MasVnrType', 'Electrical',
                   'GarageType', 'GarageFinish', 'Fence',
                   'MiscFeature']

# Features with ordinal missing data.
ord_na_features = ['BsmtQual', 'BsmtCond', 'BsmtExposure',
                   'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu',
                   'GarageQual', 'GarageCond', 'PoolQC']

'''

# Features with no need to edit.
features = ['LotArea', 'OverallQual', 'OverallCond',
            'YearBuilt', '1stFlrSF', '2ndFlrSF',
            'LowQualFinSF', 'GrLivArea', 'FullBath',
            'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
            'TotRmsAbvGrd', 'Fireplaces', 'WoodDeckSF',
            'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
            'ScreenPorch', 'PoolArea', 'MiscVal']

# Features with numerical missing data.
num_na_features = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt',
                   'BsmtFinSF1', 'TotalBsmtSF', 'GarageCars']

# Ordinal features to be mapped to 5-point score.
scoring_5_features = ['ExterQual', 'ExterCond',
                      'HeatingQC', 'KitchenQual']

# Ordinal features to be mapped to 3-point score.
scoring_3_features = ['PavedDrive']

'''
# TO DO replace this for-loop with a simple imputer, as above.
for feature in num_na_features:
   home_data[feature] = average_na(home_data[feature])    
'''

# Apply data cleaning to ordinal features.
for feature in scoring_5_features:
    home_data[feature] = score_5_ranking(home_data[feature])

for feature in scoring_3_features:
    home_data[feature] = score_3_ranking(home_data[feature])

# Combine all features together.
X = pd.concat([home_data[features], home_data[num_na_features],
               home_data[scoring_5_features], home_data[scoring_3_features]],
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
    for feature in scoring_5_features:
        test_data[feature] = score_5_ranking(test_data[feature])
    
    for feature in scoring_3_features:        
        test_data[feature] = score_3_ranking(test_data[feature])
    
    # Create test_X including only the columns you used for prediction.
    test_X = pd.concat(
        [test_data[features], test_data[num_na_features],
         test_data[scoring_5_features], test_data[scoring_3_features]],
        axis=1)
    
    # Apply SimpleImputer to model.
    impued_X, imputed_test_X = impute_X(X, test_X)
    
    # Make predictions which we will submit. 
    test_preds = rf_model_on_full_data.predict(imputed_test_X)
    
    # Save predictions in format used for competition.
    output = pd.DataFrame({'Id': test_data.Id,
                           'SalePrice': test_preds})
    
    output.to_csv('submission.csv', index=False)
    print('File successfully saved!')