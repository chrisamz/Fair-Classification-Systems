# evaluation.py

"""
Evaluation Module for Fair Classification Systems

This module contains functions for evaluating the performance and fairness of multi-class classification models
using various metrics.

Metrics Used:
- Accuracy
- Precision
- Recall
- F1-score
- Demographic parity difference
- Equalized odds difference
- Disparate impact

Libraries/Tools:
- pandas
- numpy
- scikit-learn
- AIF360
- fairlearn
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.datasets import BinaryLabelDataset
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference, MetricFrame
from fairlearn.postprocessing import ThresholdOptimizer
import joblib

class ModelEvaluation:
    def __init__(self):
        """
        Initialize the ModelEvaluation class.
        """
        pass

    def load_data(self, filepath):
        """
        Load data from a CSV file.
        
        :param filepath: str, path to the CSV file
        :return: DataFrame, loaded data
        """
        data = pd.read_csv(filepath)
        return data

    def evaluate_performance(self, y_true, y_pred):
        """
        Evaluate performance metrics for the classification model.
        
        :param y_true: array, true labels
        :param y_pred: array, predicted labels
        :return: dict, performance metrics
        """
        performance_metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted')
        }
        return performance_metrics

    def evaluate_fairness(self, X, y_true, y_pred, protected_attribute, favorable_label=1):
        """
        Evaluate fairness metrics for the classification model.
        
        :param X: DataFrame, feature matrix
        :param y_true: array, true labels
        :param y_pred: array, predicted labels
        :param protected_attribute: str, column name of the protected attribute
        :param favorable_label: int, label indicating favorable outcome (default is 1)
        :return: dict, fairness metrics
        """
        X[protected_attribute] = X[protected_attribute].astype(str)
        dataset = BinaryLabelDataset(df=X.assign(label=y_true), label_names=['label'], protected_attribute_names=[protected_attribute])
        pred_dataset = BinaryLabelDataset(df=X.assign(label=y_pred), label_names=['label'], protected_attribute_names=[protected_attribute])

        metrics = {}

        # Demographic Parity
        metric_frame = MetricFrame(metrics={"demographic_parity_difference": demographic_parity_difference}, y_true=y_true, y_pred=y_pred, sensitive_features=X[protected_attribute])
        metrics['demographic_parity_difference'] = metric_frame.overall['demographic_parity_difference']

        # Equalized Odds
        metric_frame = MetricFrame(metrics={"equalized_odds_difference": equalized_odds_difference}, y_true=y_true, y_pred=y_pred, sensitive_features=X[protected_attribute])
        metrics['equalized_odds_difference'] = metric_frame.overall['equalized_odds_difference']

        # Disparate Impact
        metric = BinaryLabelDatasetMetric(dataset, unprivileged_groups=[{protected_attribute: '0'}], privileged_groups=[{protected_attribute: '1'}])
        metrics['disparate_impact'] = metric.disparate_impact()

        return metrics

    def plot_performance_metrics(self, performance_metrics):
        """
        Plot performance metrics.
        
        :param performance_metrics: dict, performance metrics
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 6))
        plt.bar(performance_metrics.keys(), performance_metrics.values(), color=['blue', 'green', 'orange', 'red'])
        plt.title('Performance Metrics')
        plt.ylabel('Score')
        plt.show()

    def plot_fairness_metrics(self, fairness_metrics):
        """
        Plot fairness metrics.
        
        :param fairness_metrics: dict, fairness metrics
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 6))
        plt.bar(fairness_metrics.keys(), fairness_metrics.values(), color=['blue', 'green', 'orange'])
        plt.title('Fairness Metrics')
        plt.ylabel('Score')
        plt.show()

if __name__ == "__main__":
    processed_data_dir = 'data/processed/'
    X_test_filepath = f'{processed_data_dir}X_test.csv'
    y_test_filepath = f'{processed_data_dir}y_test.csv'
    y_pred_filepath = f'{processed_data_dir}y_pred.csv'
    protected_attribute = 'gender'  # Example protected attribute

    evaluator = ModelEvaluation()

    # Load test data and predictions
    X_test = evaluator.load_data(X_test_filepath)
    y_test = pd.read_csv(y_test_filepath).values.ravel()
    y_pred = pd.read_csv(y_pred_filepath).values.ravel()

    # Evaluate performance metrics
    performance_metrics = evaluator.evaluate_performance(y_test, y_pred)
    print("Performance Metrics:")
    for metric, value in performance_metrics.items():
        print(f"{metric}: {value}")

    # Plot performance metrics
    evaluator.plot_performance_metrics(performance_metrics)

    # Evaluate fairness metrics
    fairness_metrics = evaluator.evaluate_fairness(X_test, y_test, y_pred, protected_attribute)
    print("\nFairness Metrics:")
    for metric, value in fairness_metrics.items():
        print(f"{metric}: {value}")

    # Plot fairness metrics
    evaluator.plot_fairness_metrics(fairness_metrics)

    print("Model evaluation completed.")
