import os
import sys

import pandas as pd
import numpy as np
from tzuk import createLabels
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import json
import datetime
import xgboost as xgb



def get_data (csv_path):
    data = createLabels(csv_path)
    data_string = data.copy()
    payer_code_map = {'1': 'self', '2': 'mid_class_insurance', '3': 'premium'}
    data_string['payer_code'] = data['payer_code'].map(payer_code_map)
    gender_map = {'Male': 0, 'Female': 1}
    data_string['gender'] = data['gender'].map(gender_map)
    ohe = OneHotEncoder(sparse_output=False)
    transformed = ohe.fit_transform(data_string[['payer_code', 'race']])
    categories = np.hstack(ohe.categories_)
    transformed_df = pd.DataFrame(transformed, columns=categories)
    data_encoded = pd.concat(
        [data_string.drop(['race', 'payer_code', 'patient_nbr', 'encounter_id', 'race'], axis=1).reset_index(drop=True),
         transformed_df], axis=1)
    data_encoded['age_group'] = data_encoded['age_group'].astype('category').cat.codes
    data_encoded['number_diagnoses'] = data_encoded['number_diagnoses'].astype('category').cat.codes
    data_encoded['change'] = data_encoded['change'].astype('category').cat.codes
    data_encoded['diabetesMed'] = data_encoded['diabetesMed'].astype('category').cat.codes
    X = data_encoded.drop(['readmitted', 'readmitted_less_than_30', 'admission_type_id', 'admission_source_id'], axis=1)
    Y = data_encoded['readmitted_less_than_30'].astype(bool)
    test_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, stratify=Y)
    return X_train, X_test, y_train, y_test
def train_single_xgboost(X_train, y_train):
    clf = xgb.XGBClassifier(n_estimators = 50, colsample_bytree = 0.1, max_depth = 10, learning_rate=0.01)
    clf.fit(X_train, y_train)
    return clf 
    
def xgbost_matrics(clfm ,X_test, y_test):
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))

if __name__ == "__main__":
    csv_path = '.\\data\\diabetic_data.csv'
    X_train, X_test, y_train, y_test = get_data(csv_path)
    clf = train_single_xgboost(X_train, y_train)
    xgbost_matrics(clf, X_test, y_test)
