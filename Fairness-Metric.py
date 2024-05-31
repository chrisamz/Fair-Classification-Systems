# fairness_metrics.py

"""
Fairness Metrics Module for Fair Classification Systems

This module contains functions for evaluating the fairness of multi-class classification models
using various fairness metrics.

Metrics Used:
- Demographic parity
- Equal opportunity
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
from sklearn.preprocessing import LabelEncoder

class FairnessMetrics:
    def __init__(self):
        """
        Initialize the FairnessMetrics class.
        """
        self.label_encoder = LabelEncoder()

    def load_data(self, filepath):
        """
        Load data from a CSV file.
        
        :param filepath: str, path to the CSV file
        :return: DataFrame, loaded data
        """
        data = pd.read_csv(filepath)
        return data

    def encode_labels(self, data, target_column):
        """
        Encode target labels using label encoding.
        
        :param data: DataFrame, input data
        :param target_column: str, column name of the target variable
        :return: DataFrame, data with encoded target labels
        """
        data[target_column] = self.label_encoder.fit_transform(data[target_column])
        return data

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
        X[protected_attribute] = self.label_encoder.fit_transform(X[protected_attribute])
        dataset = BinaryLabelDataset(df=X, label_names=[y_true.name], protected_attribute_names=[protected_attribute])
        pred_dataset = BinaryLabelDataset(df=X.assign(pred=y_pred), label_names=['pred'], protected_attribute_names=[protected_attribute])

        metrics = {}
        
        # Demographic Parity
        metric_frame = MetricFrame(metrics={"demographic_parity_difference": demographic_parity_difference}, y_true=y_true, y_pred=y_pred, sensitive_features=X[protected_attribute])
        metrics['demographic_parity_difference'] = metric_frame.overall['demographic_parity_difference']
        
        # Equal Opportunity
        metric_frame = MetricFrame(metrics={"equalized_odds_difference": equalized_odds_difference}, y_true=y_true, y_pred=y_pred, sensitive_features=X[protected_attribute])
        metrics['equalized_odds_difference'] = metric_frame.overall['equalized_odds_difference']
        
        # Disparate Impact
        metric = BinaryLabelDatasetMetric(dataset, unprivileged_groups=[{protected_attribute: 0}], privileged_groups=[{protected_attribute: 1}])
        metrics['disparate_impact'] = metric.disparate_impact()

        return metrics

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

if __name__ == "__main__":
    X_test_filepath = 'data/processed/X_test.csv'
    y_test_filepath = 'data/processed/y_test.csv'
    y_pred_filepath = 'data/processed/y_pred.csv'
    protected_attribute = 'gender'  # Example protected attribute

    fairness_evaluator = FairnessMetrics()

    # Load test data and predictions
    X_test = fairness_evaluator.load_data(X_test_filepath)
    y_test = pd.read_csv(y_test_filepath).values.ravel()
    y_pred = pd.read_csv(y_pred_filepath).values.ravel()

    # Evaluate fairness metrics
    fairness_metrics = fairness_evaluator.evaluate_fairness(X_test, y_test, y_pred, protected_attribute)
    print("Fairness Metrics:")
    for metric, value in fairness_metrics.items():
        print(f"{metric}: {value}")

    # Evaluate performance metrics
    performance_metrics = fairness_evaluator.evaluate_performance(y_test, y_pred)
    print("\nPerformance Metrics:")
    for metric, value in performance_metrics.items():
        print(f"{metric}: {value}")

    print("Fairness and performance evaluation completed.")
