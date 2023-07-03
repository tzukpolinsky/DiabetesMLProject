import os
import sys

import pandas as pd
import numpy as np
from tzuk import createLabels
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
from sklearn import tree
import matplotlib.pyplot as plt
import json
import datetime


def usingGridSearchCV(csv_path):
    data = createLabels(csv_path)
    data_string = data.copy()
    payer_code_map = {'1': 'self', '2': 'mid_class_insurance', '3': 'premium'}
    data_string['payer_code'] = data['payer_code'].map(payer_code_map)
    gender_map = {'Male': 0, 'Female': 1}
    data_string['gender'] = data['gender'].map(gender_map)
    ohe = OneHotEncoder(sparse_output=False)
    transformed = ohe.fit_transform(data_string[['payer_code']])
    categories = np.hstack(ohe.categories_)
    transformed_df = pd.DataFrame(transformed, columns=categories)
    data_encoded = pd.concat(
        [data_string.drop(['race', 'payer_code', 'patient_nbr', 'encounter_id', 'race'], axis=1).reset_index(drop=True),
         transformed_df], axis=1)

    X = data_encoded.drop(['readmitted', 'readmitted_less_than_30', 'admission_type_id', 'admission_source_id'], axis=1)
    Y = data_encoded['readmitted_less_than_30'].astype(bool)
    param_grid = {
        'n_estimators': range(50,1000,50),
        'max_depth': list(range(5, 20)),

    }
    rf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    result_dir = os.getcwd() + "\\randomForestParamTuningResultsGridSearch"
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)
    best_score = 0
    best_params = {}
    for size in range(1, 100, 3):
        test_size = size / 100
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, stratify=Y)
        grid_search.fit(X_train, y_train)
        results_dic = {
            'score': grid_search.best_score_,
            'params': grid_search.best_params_,
            'estimator': str(grid_search.best_estimator_)
        }
        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_params = grid_search.best_params_
        print(results_dic)
        current_result_dir = result_dir + "\\testSize" + str(test_size * 100)
        if not os.path.isdir(current_result_dir):
            os.mkdir(current_result_dir)
        current_result_dir += "\\" + datetime.datetime.now().strftime("%y%m%d%H%M%S")
        if not os.path.isdir(current_result_dir):
            os.mkdir(current_result_dir)
        json_file_name = "test_size_" + str(int(test_size * 100)) + ".json"
        with open(current_result_dir + "\\" + json_file_name, "w") as f:
            json.dump(results_dic, f)
    print("best score: {}, best params: {}".format(best_score, best_params))


def manual():
    csv_path = 'C:\\Users\\tzuk9\\Documents\\dataset_diabetes\\diabetic_data.csv'
    data = createLabels(csv_path)
    data_string = data.copy()
    payer_code_map = {'1': 'self', '2': 'mid_class_insurance', '3': 'premium'}
    data_string['payer_code'] = data['payer_code'].map(payer_code_map)
    gender_map = {'Male': 0, 'Female': 1}
    data_string['gender'] = data['gender'].map(gender_map)
    ohe = OneHotEncoder(sparse_output=False)
    transformed = ohe.fit_transform(data_string[['payer_code']])
    categories = np.hstack(ohe.categories_)
    transformed_df = pd.DataFrame(transformed, columns=categories)
    data_encoded = pd.concat(
        [data_string.drop(['race', 'payer_code', 'patient_nbr', 'encounter_id', 'race'], axis=1).reset_index(drop=True),
         transformed_df], axis=1)

    X = data_encoded.drop(['readmitted', 'readmitted_less_than_30', 'admission_type_id', 'admission_source_id'], axis=1)
    Y = data_encoded['readmitted_less_than_30'].astype(bool)
    # Splitting data
    result_dir = os.getcwd() + "\\randomForestParamTuningResults"
    best_roc = 0
    best_params = {}
    for depth in range(5, 20):
        for size in range(1, 100, 3):
            test_size = size / 100
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, stratify=Y)
            # Initialize a Random Forest
            rf = RandomForestClassifier(max_depth=depth, n_jobs=-1)
            # Train the model
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            fpr, tpr, thresholds = roc_curve(y_test, y_pred)
            score = roc_auc_score(y_test, y_pred)
            if score >= best_roc:
                best_roc = score
                best_params[score] = {'depth': depth, 'test_size': test_size}
            print("test size {} depth {}".format(test_size, depth))
            report = classification_report(y_test, y_pred)
            print(report)
            current_result_dir = result_dir + "\\depth" + str(depth)
            if not os.path.isdir(current_result_dir):
                os.mkdir(current_result_dir)
            current_result_dir += "\\" + datetime.datetime.now().strftime("%y%m%d%H%M%S")
            if not os.path.isdir(current_result_dir):
                os.mkdir(current_result_dir)
            json_file_name = "test_size_" + str(int(test_size * 100)) + "_depth" + str(depth) + ".json"
            with open(current_result_dir + "\\" + json_file_name, "w") as f:
                json.dump(report, f)
            print("<++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++>")
            print("<++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++>")
            print("<++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++>")

    # estimator = rf.estimators_[6]
    #
    # # Visualize the tree using plot_tree
    # plt.figure(figsize=(15, 10))
    # tree.plot_tree(estimator,
    #                feature_names=data_encoded.columns,
    #                filled=True,
    #                impurity=True,
    #                rounded=True)
    # plt.show()


if __name__ == "__main__":
    csv_path = 'C:\\Users\\tzuk9\\Documents\\dataset_diabetes\\diabetic_data.csv'
    usingGridSearchCV(csv_path)
