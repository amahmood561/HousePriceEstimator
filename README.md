# HousePriceEstimator
 Linear Regression-implementation of a House Price Prediction model in Python 

Instructions for Running
Replace 'house_prices.csv' with the path to your actual dataset.
Adjust the features list based on the columns in your dataset.
You can extend the model by trying different regression algorithms (e.g., RandomForestRegressor) and adding hyperparameter tuning.

# how to use implementation

# Define the features, target, numeric, and categorical features
features = ['Size', 'Bedrooms', 'Age', 'Location']
target = 'Price'
numeric_features = ['Size', 'Bedrooms', 'Age']
categorical_features = ['Location']

# Create an instance of the HousePricePredictor class
predictor = HousePricePredictor(features, target, numeric_features, categorical_features)

# Run the entire prediction pipeline
predictor.run('house_prices.csv')


