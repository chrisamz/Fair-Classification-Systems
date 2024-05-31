# fairness_constraints.py

"""
Fairness Constraints Module for Fair Classification Systems

This module contains functions for applying fairness constraints to multi-class classification models
to ensure fairness across different demographic groups.

Techniques Used:
- Post-processing methods
- In-processing methods
- Pre-processing methods

Libraries/Tools:
- pandas
- numpy
- scikit-learn
- AIF360
- fairlearn
"""

import pandas as pd
import numpy as np
from aif360.algorithms.postprocessing import EqOddsPostprocessing
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.inprocessing import AdversarialDebiasing
from aif360.datasets import BinaryLabelDataset
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from sklearn.linear_model import LogisticRegression
import tensorflow as tf

class FairnessConstraints:
    def __init__(self):
        """
        Initialize the FairnessConstraints class.
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

    def apply_preprocessing(self, X_train, y_train, protected_attribute):
        """
        Apply pre-processing methods to ensure fairness.
        
        :param X_train: DataFrame, training feature matrix
        :param y_train: Series, training target vector
        :param protected_attribute: str, column name of the protected attribute
        :return: DataFrame, DataFrame, reweighted training features and labels
        """
        X_train[protected_attribute] = X_train[protected_attribute].astype(str)
        dataset = BinaryLabelDataset(df=pd.concat([X_train, y_train], axis=1), label_names=[y_train.name], protected_attribute_names=[protected_attribute])
        reweigher = Reweighing()
        reweighted_dataset = reweigher.fit_transform(dataset)

        X_train_reweighted = pd.DataFrame(reweighted_dataset.features, columns=X_train.columns)
        y_train_reweighted = pd.Series(reweighted_dataset.labels.ravel(), name=y_train.name)
        return X_train_reweighted, y_train_reweighted

    def apply_inprocessing(self, X_train, y_train, protected_attribute):
        """
        Apply in-processing methods to ensure fairness.
        
        :param X_train: DataFrame, training feature matrix
        :param y_train: Series, training target vector
        :param protected_attribute: str, column name of the protected attribute
        :return: model, trained adversarial debiasing model
        """
        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()
        X_train[protected_attribute] = X_train[protected_attribute].astype(str)
        dataset = BinaryLabelDataset(df=pd.concat([X_train, y_train], axis=1), label_names=[y_train.name], protected_attribute_names=[protected_attribute])

        model = AdversarialDebiasing(privileged_groups=[{protected_attribute: '1'}], unprivileged_groups=[{protected_attribute: '0'}], scope_name='debiasing', sess=sess)
        model.fit(dataset)
        return model

    def apply_postprocessing(self, X_train, y_train, X_test, y_test, protected_attribute):
        """
        Apply post-processing methods to ensure fairness.
        
        :param X_train: DataFrame, training feature matrix
        :param y_train: Series, training target vector
        :param X_test: DataFrame, testing feature matrix
        :param y_test: Series, testing target vector
        :param protected_attribute: str, column name of the protected attribute
        :return: array, post-processed predictions
        """
        lr = LogisticRegression(solver='lbfgs', max_iter=1000)
        lr.fit(X_train, y_train)
        y_pred_prob = lr.predict_proba(X_test)[:, 1]
        
        dataset_test = BinaryLabelDataset(df=pd.concat([X_test, y_test], axis=1), label_names=[y_test.name], protected_attribute_names=[protected_attribute])
        dataset_pred = BinaryLabelDataset(df=pd.concat([X_test, pd.Series(y_pred_prob, name='pred_prob')], axis=1), label_names=['pred_prob'], protected_attribute_names=[protected_attribute])

        eq_odds = EqOddsPostprocessing(unprivileged_groups=[{protected_attribute: 0}], privileged_groups=[{protected_attribute: 1}])
        eq_odds = eq_odds.fit(dataset_test, dataset_pred)
        postprocessed_dataset = eq_odds.transform(dataset_pred)

        y_pred_postprocessed = postprocessed_dataset.labels.ravel()
        return y_pred_postprocessed

    def apply_fairlearn_postprocessing(self, X_train, y_train, X_test, y_test, protected_attribute):
        """
        Apply Fairlearn's post-processing method (ThresholdOptimizer) to ensure fairness.
        
        :param X_train: DataFrame, training feature matrix
        :param y_train: Series, training target vector
        :param X_test: DataFrame, testing feature matrix
        :param y_test: Series, testing target vector
        :param protected_attribute: str, column name of the protected attribute
        :return: array, post-processed predictions
        """
        lr = LogisticRegression(solver='lbfgs', max_iter=1000)
        lr.fit(X_train, y_train)
        y_pred_prob = lr.predict_proba(X_test)[:, 1]

        threshold_optimizer = ThresholdOptimizer(
            estimator=lr,
            constraints="demographic_parity",
            objective="accuracy_score",
            prefit=True
        )
        threshold_optimizer.fit(X_train, y_train, sensitive_features=X_train[protected_attribute])
        y_pred_postprocessed = threshold_optimizer.predict(X_test, sensitive_features=X_test[protected_attribute])
        return y_pred_postprocessed

if __name__ == "__main__":
    processed_data_dir = 'data/processed/'
    X_train_filepath = f'{processed_data_dir}X_train.csv'
    y_train_filepath = f'{processed_data_dir}y_train.csv'
    X_test_filepath = f'{processed_data_dir}X_test.csv'
    y_test_filepath = f'{processed_data_dir}y_test.csv'
    protected_attribute = 'gender'  # Example protected attribute

    fairness_constraints = FairnessConstraints()

    # Load preprocessed data
    X_train = fairness_constraints.load_data(X_train_filepath)
    y_train = pd.read_csv(y_train_filepath).values.ravel()
    X_test = fairness_constraints.load_data(X_test_filepath)
    y_test = pd.read_csv(y_test_filepath).values.ravel()

    # Apply pre-processing method
    X_train_reweighted, y_train_reweighted = fairness_constraints.apply_preprocessing(X_train, y_train, protected_attribute)
    print("Pre-processing applied.")

    # Apply in-processing method
    adversarial_model = fairness_constraints.apply_inprocessing(X_train, y_train, protected_attribute)
    print("In-processing applied with adversarial debiasing.")

    # Apply post-processing method using AIF360
    y_pred_postprocessed_aif360 = fairness_constraints.apply_postprocessing(X_train, y_train, X_test, y_test, protected_attribute)
    print("Post-processing applied using AIF360.")

    # Apply post-processing method using Fairlearn
    y_pred_postprocessed_fairlearn = fairness_constraints.apply_fairlearn_postprocessing(X_train, y_train, X_test, y_test, protected_attribute)
    print("Post-processing applied using Fairlearn.")

    print("Fairness constraints applied.")
