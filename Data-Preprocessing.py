# data_preprocessing.py

"""
Data Preprocessing Module for Fair Classification Systems

This module contains functions for collecting, cleaning, normalizing, and preparing
data for building fair multi-class classification models.

Techniques Used:
- Data cleaning
- Normalization
- Feature extraction
- Handling missing data

Libraries/Tools:
- pandas
- numpy
- scikit-learn
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

class DataPreprocessing:
    def __init__(self):
        """
        Initialize the DataPreprocessing class.
        """
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.encoder = OneHotEncoder(sparse=False)

    def load_data(self, filepath):
        """
        Load data from a CSV file.
        
        :param filepath: str, path to the CSV file
        :return: DataFrame, loaded data
        """
        data = pd.read_csv(filepath)
        return data

    def clean_data(self, data):
        """
        Clean the data by removing duplicates and handling missing values.
        
        :param data: DataFrame, input data
        :return: DataFrame, cleaned data
        """
        data = data.drop_duplicates()
        data = pd.DataFrame(self.imputer.fit_transform(data), columns=data.columns)
        return data

    def encode_categorical_features(self, data, categorical_columns):
        """
        Encode categorical features using one-hot encoding.
        
        :param data: DataFrame, input data
        :param categorical_columns: list, column names of categorical features
        :return: DataFrame, data with encoded categorical features
        """
        categorical_data = data[categorical_columns]
        encoded_data = self.encoder.fit_transform(categorical_data)
        encoded_columns = self.encoder.get_feature_names_out(categorical_columns)
        encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns)
        
        data = data.drop(categorical_columns, axis=1)
        data = pd.concat([data, encoded_df], axis=1)
        return data

    def normalize_data(self, data, numerical_columns):
        """
        Normalize the numerical features using standard scaling.
        
        :param data: DataFrame, input data
        :param numerical_columns: list, column names of numerical features
        :return: DataFrame, normalized data
        """
        data[numerical_columns] = self.scaler.fit_transform(data[numerical_columns])
        return data

    def split_data(self, data, target_column, test_size=0.2, random_state=42):
        """
        Split the data into training and testing sets.
        
        :param data: DataFrame, input data
        :param target_column: str, column name of the target variable
        :param test_size: float, proportion of the dataset to include in the test split
        :param random_state: int, random state for reproducibility
        :return: DataFrame, DataFrame, Series, Series, training and testing sets
        """
        X = data.drop(target_column, axis=1)
        y = data[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test

    def preprocess(self, filepath, categorical_columns, numerical_columns, target_column):
        """
        Execute the full preprocessing pipeline.
        
        :param filepath: str, path to the input data file
        :param categorical_columns: list, column names of categorical features
        :param numerical_columns: list, column names of numerical features
        :param target_column: str, column name of the target variable
        :return: DataFrame, DataFrame, Series, Series, training and testing sets
        """
        data = self.load_data(filepath)
        data = self.clean_data(data)
        data = self.encode_categorical_features(data, categorical_columns)
        data = self.normalize_data(data, numerical_columns)
        X_train, X_test, y_train, y_test = self.split_data(data, target_column)
        return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    raw_data_filepath = 'data/raw/data.csv'
    processed_data_dir = 'data/processed/'
    categorical_columns = ['gender', 'race', 'education']  # Example categorical features
    numerical_columns = ['age', 'income', 'experience']  # Example numerical features
    target_column = 'target'  # Example target variable

    preprocessing = DataPreprocessing()

    # Preprocess the data
    X_train, X_test, y_train, y_test = preprocessing.preprocess(
        raw_data_filepath, categorical_columns, numerical_columns, target_column
    )

    # Save the preprocessed data
    X_train.to_csv(f'{processed_data_dir}X_train.csv', index=False)
    X_test.to_csv(f'{processed_data_dir}X_test.csv', index=False)
    y_train.to_csv(f'{processed_data_dir}y_train.csv', index=False)
    y_test.to_csv(f'{processed_data_dir}y_test.csv', index=False)
    print("Data preprocessing completed and saved to 'data/processed/'.")
