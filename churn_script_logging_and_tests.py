"""Test churn_library"""
# Author: Mykola
# Created: 10.02.2022

import logging
import os

import numpy as np

from churn_library import import_data, perform_eda, encoder_helper, \
    perform_feature_engineering, train_models

# import churn_library_solution as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    """
    test data import - this example is completed for you to assist with the other test functions
    """
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda, df_test):
    """
    test perform eda function
    """
    perform_eda(df_test)

    try:
        assert len(os.listdir("./images/eda")) == 5
    except AssertionError as err:
        logging.error('Testing perform_eda: missing images.')
        raise err
    logging.info('Testing perform_eda: SUCCESS')


def test_encoder_helper(encoder_helper, df_test):
    """
    test encoder helper
    """
    df_test = encoder_helper(df_test, category_lst=cat_columns, response=RESPONSE)

    try:
        assert set(df_test.columns).issuperset([col + RESPONSE for col in cat_columns])
    except AssertionError:
        logging.error('Testing test_encoder_helper: missing categorical columns.')
    logging.info('Testing test_encoder_helper: SUCCESS')


def test_perform_feature_engineering(perform_feature_engineering, df_test):
    """
    test perform_feature_engineering
    """
    x_train, x_test, y_train, y_test = perform_feature_engineering(df_test)

    try:
        assert (len(x_train) == len(y_train)) and (len(x_test)) == len(y_test)
    except AssertionError:
        logging.error('Testing perform_feature_engineering: wrong shape.')
    logging.info('Testing perform_feature_engineering: SUCCESS')


def test_train_models(train_models, x_train, x_test, y_train, y_test):
    """
    test train_models
    """
    train_models(x_train, x_test, y_train, y_test)

    try:
        assert len(os.listdir("./models")) == 2
    except AssertionError as err:
        logging.error('Testing train_models: missing models.')
        raise err
    logging.info('Testing train_models: SUCCESS')


if __name__ == "__main__":
    test_import(import_data)

    df = import_data("./data/bank_data.csv")
    df_test = df.iloc[:100]
    df_test['Churn'] = np.where(df_test['Attrition_Flag'] == 'Existing Customer', 0, 1)

    test_eda(perform_eda, df_test)
    RESPONSE = '_Churn'
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    test_encoder_helper(encoder_helper, df_test)
    df_test = encoder_helper(df_test, category_lst=cat_columns, response=RESPONSE)

    test_perform_feature_engineering(perform_feature_engineering, df_test)
    x_train, x_test, y_train, y_test = perform_feature_engineering(df_test)

    test_train_models(train_models, x_train, x_test, y_train, y_test)
