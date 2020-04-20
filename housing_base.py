import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

'''

Base model score: 21122.78475

'''

# True - testing against train set to measure accuracy.
# False - testing against test set and save for submission.
is_accuracy = True

# Path of the file to read.
iowa_file_path = 'train.csv'
home_data = pd.read_csv(iowa_file_path)

# Create target object and call it y
y = home_data.SalePrice

# Create X
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF',
            'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[features]

if is_accuracy:
    
    # Split into validation and training data.
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
    
    # Specify model.
    iowa_model = DecisionTreeRegressor(random_state=1)
    
    # Fit model.
    iowa_model.fit(train_X, train_y)
    
    # Make validation predictions and calculate mean absolute error.
    val_predictions = iowa_model.predict(val_X)
    val_mae = mean_absolute_error(val_predictions, val_y)
    print("Validation MAE when not specifying base max_leaf_nodes: {:,.0f}".
          format(val_mae))
    
    # Using best value for max_leaf_nodes
    iowa_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
    iowa_model.fit(train_X, train_y)
    val_predictions = iowa_model.predict(val_X)
    val_mae = mean_absolute_error(val_predictions, val_y)
    print("Validation MAE for best value of base max_leaf_nodes: {:,.0f}".
          format(val_mae))
    
    # Define the model. Set random_state to 1
    rf_model = RandomForestRegressor(random_state=1)
    rf_model.fit(train_X, train_y)
    rf_val_predictions = rf_model.predict(val_X)
    rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)
    
    print("Validation MAE for base Random Forest Model: {:,.0f}".
          format(rf_val_mae))
    
else:
        
    # Create a new Random Forest model.
    rf_model_on_full_data = RandomForestRegressor()
    
    # fit rf_model_on_full_data on all data from the training data
    rf_model_on_full_data.fit(X, y)
    
    # path to file you will use for predictions
    test_data_path = 'test.csv'
    
    # read test data file using pandas
    test_data = pd.read_csv(test_data_path)
    
    # create test_X which comes from test_data but includes only the columns you used for prediction.
    # The list of columns is stored in a variable called features
    test_X = test_data[features]
    
    # make predictions which we will submit. 
    test_preds = rf_model_on_full_data.predict(test_X)
    
    # Save predictions in format used for competition.
    output = pd.DataFrame({'Id': test_data.Id,
                           'SalePrice': test_preds})
    
    output.to_csv('submission.csv', index=False)