# multi_class_classification.py

"""
Multi-Class Classification Module for Fair Classification Systems

This module contains functions for building, training, and evaluating multi-class classification models.
The models are trained to ensure fairness and high performance.

Techniques Used:
- Logistic Regression
- Decision Trees
- Random Forest
- Neural Networks

Libraries/Tools:
- pandas
- numpy
- scikit-learn
- TensorFlow
- joblib
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import joblib

class MultiClassClassification:
    def __init__(self):
        """
        Initialize the MultiClassClassification class.
        """
        self.models = {
            'logistic_regression': LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200),
            'decision_tree': DecisionTreeClassifier(),
            'random_forest': RandomForestClassifier(n_estimators=100),
            'neural_network': MLPClassifier(hidden_layer_sizes=(100,), max_iter=300)
        }
        self.keras_model = None

    def load_data(self, filepath):
        """
        Load data from a CSV file.
        
        :param filepath: str, path to the CSV file
        :return: DataFrame, loaded data
        """
        data = pd.read_csv(filepath)
        return data

    def train_model(self, X_train, y_train, model_name):
        """
        Train a multi-class classification model.
        
        :param X_train: DataFrame, training feature matrix
        :param y_train: Series, training target vector
        :param model_name: str, name of the model to train
        """
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} is not defined.")
        model.fit(X_train, y_train)
        joblib.dump(model, f'models/{model_name}_model.pkl')
        print(f"{model_name} model trained and saved.")

    def train_keras_model(self, X_train, y_train, num_classes):
        """
        Train a neural network model using Keras.
        
        :param X_train: DataFrame, training feature matrix
        :param y_train: Series, training target vector
        :param num_classes: int, number of classes in the target variable
        """
        y_train = to_categorical(y_train, num_classes)
        self.keras_model = Sequential()
        self.keras_model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
        self.keras_model.add(Dense(64, activation='relu'))
        self.keras_model.add(Dense(num_classes, activation='softmax'))
        
        self.keras_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.keras_model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=1)
        self.keras_model.save('models/keras_model.h5')
        print("Keras neural network model trained and saved.")

    def evaluate_model(self, X_test, y_test, model_name):
        """
        Evaluate a multi-class classification model.
        
        :param X_test: DataFrame, testing feature matrix
        :param y_test: Series, true labels for testing
        :param model_name: str, name of the model to evaluate
        """
        model = joblib.load(f'models/{model_name}_model.pkl')
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        print(f"{model_name} model evaluation:")
        for metric, value in metrics.items():
            print(f"{metric}: {value}")

    def evaluate_keras_model(self, X_test, y_test, num_classes):
        """
        Evaluate the neural network model trained using Keras.
        
        :param X_test: DataFrame, testing feature matrix
        :param y_test: Series, true labels for testing
        :param num_classes: int, number of classes in the target variable
        """
        y_test = to_categorical(y_test, num_classes)
        keras_model = Sequential()
        keras_model = keras.models.load_model('models/keras_model.h5')
        _, accuracy = keras_model.evaluate(X_test, y_test, verbose=1)
        print(f"Keras neural network model accuracy: {accuracy}")

if __name__ == "__main__":
    processed_data_dir = 'data/processed/'
    X_train_filepath = f'{processed_data_dir}X_train.csv'
    X_test_filepath = f'{processed_data_dir}X_test.csv'
    y_train_filepath = f'{processed_data_dir}y_train.csv'
    y_test_filepath = f'{processed_data_dir}y_test.csv'

    classification = MultiClassClassification()

    # Load preprocessed data
    X_train = classification.load_data(X_train_filepath)
    X_test = classification.load_data(X_test_filepath)
    y_train = pd.read_csv(y_train_filepath).values.ravel()
    y_test = pd.read_csv(y_test_filepath).values.ravel()
    num_classes = len(np.unique(y_train))

    # Train and evaluate logistic regression model
    classification.train_model(X_train, y_train, 'logistic_regression')
    classification.evaluate_model(X_test, y_test, 'logistic_regression')

    # Train and evaluate decision tree model
    classification.train_model(X_train, y_train, 'decision_tree')
    classification.evaluate_model(X_test, y_test, 'decision_tree')

    # Train and evaluate random forest model
    classification.train_model(X_train, y_train, 'random_forest')
    classification.evaluate_model(X_test, y_test, 'random_forest')

    # Train and evaluate neural network model
    classification.train_model(X_train, y_train, 'neural_network')
    classification.evaluate_model(X_test, y_test, 'neural_network')

    # Train and evaluate Keras neural network model
    classification.train_keras_model(X_train, y_train, num_classes)
    classification.evaluate_keras_model(X_test, y_test, num_classes)

    print("Multi-class classification modeling completed.")
