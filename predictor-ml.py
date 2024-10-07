import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
'''

Explanation
__init__: Initializes the class with features, target, numeric, and categorical features. It also creates the model pipeline.
_create_pipeline: Private method to create a preprocessing and model pipeline.
load_data: Loads the dataset from a specified file path and displays the first few rows.
preprocess: Splits the data into features (X) and target (y).
train_test_split: Splits the data into training and testing sets.
train: Trains the model on the training set.
predict: Makes predictions using the test set.
evaluate: Evaluates the model using mean absolute error, mean squared error, and root mean squared error.
run: Executes the whole process in sequence.


'''
class HousePricePredictor:
    def __init__(self, features, target, numeric_features, categorical_features):
        self.features = features
        self.target = target
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.model = self._create_pipeline()
    
    def _create_pipeline(self):
        # Create transformers for numerical and categorical data
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])
        
        # Combine the preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)])
        
        # Define the model pipeline
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())])
        
        return model

    def load_data(self, file_path):
        # Load the dataset
        self.data = pd.read_csv(file_path)
        print(self.data.head())
    
    def preprocess(self):
        # Split into features (X) and target (y)
        self.X = self.data[self.features]
        self.y = self.data[self.target]
    
    def train_test_split(self, test_size=0.2, random_state=42):
        # Split the data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state)
    
    def train(self):
        # Fit the model to the training data
        self.model.fit(self.X_train, self.y_train)
    
    def predict(self):
        # Make predictions on the test set
        self.y_pred = self.model.predict(self.X_test)
    
    def evaluate(self):
        # Evaluate the model
        mae = mean_absolute_error(self.y_test, self.y_pred)
        mse = mean_squared_error(self.y_test, self.y_pred)
        rmse = np.sqrt(mse)
        print(f'Mean Absolute Error: {mae}')
        print(f'Mean Squared Error: {mse}')
        print(f'Root Mean Squared Error: {rmse}')

    def run(self, file_path):
        self.load_data(file_path)
        self.preprocess()
        self.train_test_split()
        self.train()
        self.predict()
        self.evaluate()
