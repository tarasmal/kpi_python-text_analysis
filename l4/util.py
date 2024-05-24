import pandas as pd
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


def read_csv_data(path_to_file="./news.csv"):
    data = pd.read_csv(path_to_file)
    return data


def prepare_data_train_test_data(data: DataFrame, test_size=0.4):
    X = data["text"]
    y = data["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

