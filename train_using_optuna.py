"""
Optuna example that optimizes a classifier configuration for cancer dataset using LightGBM.

In this example, we optimize the validation accuracy of cancer detection using LightGBM.
We optimize both the choice of booster model and their hyperparameters.

"""
import datetime
import os.path
import sys

import numpy as np
import optuna

import lightgbm as lgb
import sklearn.datasets
import sklearn.metrics
from imblearn.ensemble import BalancedBaggingClassifier, BalancedRandomForestClassifier
from sklearn.model_selection import train_test_split
from utils import createLabels
import xgboost as xgb

X = []
Y = []
TEST_SIZE = 0.5


def objectiveSVC(trial):
    global TEST_SIZE
    global X
    global Y
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=TEST_SIZE)

    svc_c = trial.suggest_float("svc_c", 1e-10, 1e10, log=True)
    classifier_obj = sklearn.svm.SVC(C=svc_c, gamma="auto")
    classifier_obj.fit(X_train, y_train)
    y_pred = classifier_obj.predict(X_test)
    accuracy = sklearn.metrics.recall_score(y_test, y_pred)
    return accuracy


def objectiveBalancedRandomForestClassifier(trail):
    global TEST_SIZE
    global X
    global Y
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=TEST_SIZE)
    params = {
        'n_estimators': trail.suggest_categorical('n_estimators', list(range(50, 400, 50))),
        'max_depth': trail.suggest_int('max_depth', 2, 30),
        'sampling_strategy': trail.suggest_categorical('sampling_strategy', [0.5, 'not minority', 'all'])
    }
    classifier_obj = BalancedRandomForestClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'],
                                                    replacement=True, sampling_strategy=params['sampling_strategy'])
    classifier_obj.fit(X_train, y_train)
    y_pred = classifier_obj.predict(X_test)

    accuracy = sklearn.metrics.recall_score(y_test, y_pred)
    return accuracy


def objectiveBalancedBaggingClassifier(trail):
    global TEST_SIZE
    global X
    global Y
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=TEST_SIZE)
    params = {
        'n_estimators': trail.suggest_categorical('n_estimators', list(range(50, 400, 50))),
        'sampling_strategy': trail.suggest_categorical('sampling_strategy', [0.5, 'not minority', 'all'])
    }
    classifier_obj = BalancedBaggingClassifier(replacement=True, sampling_strategy=params['sampling_strategy'],
                                               n_estimators=params['n_estimators'])
    classifier_obj.fit(X_train, y_train)
    y_pred = classifier_obj.predict(X_test)

    accuracy = sklearn.metrics.recall_score(y_test, y_pred)
    return accuracy


def objectiveRandomForest(trial):
    global TEST_SIZE
    global X
    global Y
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=TEST_SIZE)
    params = {
        'n_estimators': trial.suggest_categorical('n_estimators', list(range(50, 400, 50))),
        'max_depth': trial.suggest_int('max_depth',2,30),
    }
    classifier_obj = sklearn.ensemble.RandomForestClassifier(max_depth=params['max_depth'],n_estimators=params['n_estimators'])
    classifier_obj.fit(X_train, y_train)
    y_pred = classifier_obj.predict(X_test)

    accuracy = sklearn.metrics.recall_score(y_test, y_pred)
    return accuracy


def objectiveXgboost(trial):
    global TEST_SIZE
    global X
    global Y
    train_x, valid_x, train_y, valid_y = train_test_split(X, Y, test_size=TEST_SIZE)
    dtrain = xgb.DMatrix(train_x, label=train_y)
    dvalid = xgb.DMatrix(valid_x, label=valid_y)

    param = {
        "verbosity": 0,
        "objective": "binary:logistic",
        # use exact for small dataset.
        "tree_method": "exact",
        # defines booster, gblinear for linear functions.
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        # L2 regularization weight.
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        # L1 regularization weight.
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        # sampling ratio for training data.
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        # sampling according to each tree.
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
    }

    if param["booster"] in ["gbtree", "dart"]:
        # maximum depth of the tree, signifies complexity of the tree.
        param["max_depth"] = trial.suggest_int("max_depth", 3, 9, step=2)
        # minimum child weight, larger the term more conservative the tree.
        param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
        param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        # defines how selective algorithm is.
        param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

    bst = xgb.train(param, dtrain)
    preds = bst.predict(dvalid)
    pred_labels = np.rint(preds)
    accuracy = sklearn.metrics.recall_score(valid_y, pred_labels)
    return accuracy


def objectiveLGBM(trial):
    global TEST_SIZE
    global X
    global Y
    train_x, valid_x, train_y, valid_y = train_test_split(X, Y, test_size=TEST_SIZE)
    dtrain = lgb.Dataset(train_x, label=train_y)

    param = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }

    gbm = lgb.train(param, dtrain)
    preds = gbm.predict(valid_x)
    pred_labels = np.rint(preds)
    accuracy = sklearn.metrics.recall_score(valid_y, pred_labels)
    return accuracy


if __name__ == "__main__":
    data = createLabels(sys.argv[1])
    data_encoded = data.copy()
    X = data_encoded.drop(['readmitted', 'readmitted_less_than_30', 'encounter_id', 'patient_nbr'], axis=1)
    Y = data_encoded['readmitted_less_than_30'].astype(bool)
    functions = [[objectiveLGBM, 'lgbm'], [objectiveXgboost, 'xgboost'],
                 [objectiveRandomForest, 'RandomForest'],
                 [objectiveBalancedBaggingClassifier, "BalancedBaggingClassifier"],
                 [objectiveBalancedRandomForestClassifier, "BalancedRandomForestClassifier"]]
    result_path = sys.argv[2]
    if not os.path.isdir(result_path):
        os.mkdir(result_path)
    result_path = os.path.join(result_path,datetime.datetime.now().strftime("%y%m%d%H%M%S"))
    os.mkdir(result_path)
    for objective, study_name in functions:

        study = optuna.create_study(direction="maximize", study_name=study_name)
        study.optimize(objective, n_trials=300, n_jobs=-1, show_progress_bar=True)
        with open(result_path + "/" + study_name, "w") as file:
            print("Number of finished trials for {}: {}".format(study_name,len(study.trials)))
            file.write("Number of finished trials: {}\n".format(len(study.trials)))

            print("Best trial:")
            file.write("Best trial:\n")
            trial = study.best_trial

            print("  Value: {}".format(trial.value))
            file.write("  Value: {}\n".format(trial.value))

            print("  Params: ")
            file.write("  Params: \n")
            for key, value in trial.params.items():
                print("    {}: {}".format(key, value))
                file.write("    {}: {}\n".format(key, value))
