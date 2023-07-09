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
import xgboost as xgb


def usingGridSearchCV(csv_path, result_path):
    data = createLabels(csv_path)
    data_encoded = data.copy()
    data_encoded['age_group'] = data_encoded['age_group'].astype('category')
    data_encoded['number_diagnoses'] = data_encoded['number_diagnoses'].astype('category')
    data_encoded['change'] = data_encoded['change'].astype('category')
    data_encoded['diabetesMed'] = data_encoded['diabetesMed'].astype('category')
    data_encoded['diabetesMed'] = data_encoded['diabetesMed'].astype('category')
    data_encoded['admission_source_id'] = data_encoded['admission_source_id'].astype(int)
    data_encoded['admission_type_id'] = data_encoded['admission_type_id'].astype(int)
    data_encoded['num_medications'] = data_encoded['num_medications'].astype(int)
    data_encoded['payer_code'] = data_encoded['payer_code'].astype(int)
    data_encoded['discharge_disposition_id'] = data_encoded['discharge_disposition_id'].astype(int)
    X = data_encoded.drop(['readmitted', 'readmitted_less_than_30', 'encounter_id', 'patient_nbr'], axis=1)
    Y = data_encoded['readmitted_less_than_30'].astype(bool)
    param_grid = {
        'n_estimators': list(range(50, 400, 50)),
        'max_depth': list(range(2, 15)),
        'tree_method': ['gpu_hist'],
        'enable_categorical':[True],
        'subsample': [0.3,0.5,0.7,1],
        'max_bin':[256,512,1024]
    }
    rf = xgb.XGBClassifier()
    result_dir = os.path.join(result_path, "randomForestParamTuningResultsGridSearch")
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)
    best_score = 0
    best_params = {}
    best_report = None
    for size in range(1, 100, 10):
        test_size = size / 100
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='recall', n_jobs=4)

        grid_search.fit(X_train, y_train)

        y_pred = grid_search.predict(X_test)
        report = classification_report(y_test, y_pred)
        print(report)
        roc_score = roc_auc_score(y_test, y_pred)
        print("roc score: {}".format(roc_score))
        results_dic = {
            'grid_search_score': grid_search.best_score_,
            'params': grid_search.best_params_,
            'estimator': str(grid_search.best_estimator_),
            'report': str(report),
            'roc_score': str(roc_score)
        }
        if roc_score > best_score:
            best_score = roc_score
            best_params = grid_search.best_params_
            best_report = report
        print("test size: {} grid_search.best_params_: {}".format(int(test_size * 100), grid_search.best_params_))
        current_result_dir = os.path.join(result_dir, "testSize" + str(int(test_size * 100)))
        if not os.path.isdir(current_result_dir):
            os.mkdir(current_result_dir)
        current_result_dir = os.path.join(current_result_dir, datetime.datetime.now().strftime("%y%m%d%H%M%S"))
        if not os.path.isdir(current_result_dir):
            os.mkdir(current_result_dir)
        json_file_name = "test_size_" + str(int(test_size * 100)) + ".json"
        with open(os.path.join(current_result_dir, json_file_name), "w") as f:
            json.dump(results_dic, f)
    print("best score: {}, best params: {}".format(best_score, best_params))
    print(best_report)


def manual(csv_path, result_path):
    data = createLabels(csv_path)
    data_encoded = data.copy()
    data_encoded['age_group'] = data_encoded['age_group'].astype('category')
    data_encoded['number_diagnoses'] = data_encoded['number_diagnoses'].astype('category')
    data_encoded['change'] = data_encoded['change'].astype('category')
    data_encoded['diabetesMed'] = data_encoded['diabetesMed'].astype('category')
    data_encoded['diabetesMed'] = data_encoded['diabetesMed'].astype('category')
    data_encoded['admission_source_id'] = data_encoded['admission_source_id'].astype(int)
    data_encoded['admission_type_id'] = data_encoded['admission_type_id'].astype(int)
    data_encoded['num_medications'] = data_encoded['num_medications'].astype(int)
    data_encoded['payer_code'] = data_encoded['payer_code'].astype(int)
    data_encoded['discharge_disposition_id'] = data_encoded['discharge_disposition_id'].astype(int)
    X = data_encoded.drop(['readmitted', 'readmitted_less_than_30','encounter_id','patient_nbr'], axis=1)
    Y = data_encoded['readmitted_less_than_30'].astype(bool)
    # Splitting data
    result_dir = os.path.join(result_path, "randomForestParamTuningResults")
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)
    best_roc = 0
    best_params = {}
    for depth in range(5, 30):
        for size in range(1, 100, 10):
            test_size = size / 100
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)
            # Initialize a Random Forest
            #rf = RandomForestClassifier(max_depth=depth, n_jobs=-1, max_features='auto', n_estimators=500)
            rf = xgb.XGBClassifier(n_estimators = 350,tree_method="gpu_hist", max_depth = depth,enable_categorical=True)
            # Train the model
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            score = roc_auc_score(y_test, y_pred)
            print("test size {} depth {}".format(test_size, depth))
            if score > best_roc:
                best_roc = score
                best_params[score] = {'depth': depth, 'test_size': test_size}
                report = classification_report(y_test, y_pred)
                print(report)
                print(score)
                current_result_dir = os.path.join(result_dir, "depth" + str(depth))
                if not os.path.isdir(current_result_dir):
                    os.mkdir(current_result_dir)
                current_result_dir = os.path.join(current_result_dir, str(int(100 * test_size)))
                if not os.path.isdir(current_result_dir):
                    os.mkdir(current_result_dir)
                json_file_name = datetime.datetime.now().strftime("%y%m%d%H%M%S") + ".json"
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
    csv_path = sys.argv[1]
    result_path = sys.argv[2]
    # manual(csv_path, result_path)
    usingGridSearchCV(csv_path, result_path)
