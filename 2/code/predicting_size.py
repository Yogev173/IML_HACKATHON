####### IMPORTS #######
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb

from predicting_metastases_filter import filter_data


#######################

def load_data(file_path):
    """
    Loads data from a CSV file and returns a pandas DataFrame.
    """
    data = pd.read_csv(file_path)
    return data



def fit_and_classify(X_train, y_train):
    """
    Fits a multilabel classification model on the training data and performs classification on the test data.
    Returns classification report.
    """
    model = OneVsRestClassifier(RandomForestRegressor())

    model.fit(X_train, y_train)
    return model


def train_xgboost_classifier_model(X_train, y_train):
    # Map labels 'A', 'B', 'C', 'D' to integer values
    label_mapping = {'LYM - Lymph nodes': 0, 'PUL - Pulmonary': 1, 'BON - Bones': 2, 'SKI - Skin': 3,
                     'HEP - Hepatic': 4}
    y_train_mapped = y_train.apply(lambda labels: [label_mapping[label] for label in labels])

    # Preprocess the target variable using MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    y_train_encoded = mlb.fit_transform(y_train_mapped)

    model = xgb.XGBClassifier(objective='binary:logistic')
    model.fit(X_train, y_train_encoded)


def train_xgboost_regressor_model(X_train, y_train):
    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)

    return model

def main():
    # Step 1: Load the data
    file_path = "../data/train.feats.csv"
    data = load_data(file_path)

    numercial_data = filter_data(data)

    # for cat in numercial_data.keys():
    #     print(f"{cat}: {max(numercial_data[cat])}")
    numercial_data.to_csv('temp_data.csv')

    from sklearn.impute import KNNImputer

    imputer = KNNImputer(n_neighbors=4)
    numercial_data = imputer.fit_transform(numercial_data)
    numercial_data.to_csv('../data/train_feats_filtered.csv')

    y_train = load_data("../data/train.labels.1.csv")

    # Step 5: Fit the classifier and perform classification
    # model = fit_and_classify(numercial_data, y_train)
    model = xgb.XGBRegressor()
    model.fit(numercial_data, y_train)

    # Example usage
    model.



if __name__ == "__main__":
    main()
