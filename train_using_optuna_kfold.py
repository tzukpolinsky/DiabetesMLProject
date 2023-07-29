"""
Optuna example that optimizes a classifier configuration for cancer dataset using LightGBM.

In this example, we optimize the validation accuracy of cancer detection using LightGBM.
We optimize both the choice of booster model and their hyperparameters.

"""
#pylint: disable=W0602
import datetime
import os.path
import sys

import numpy as np
import optuna
from os.path import join
import lightgbm as lgb
import sklearn.datasets
import sklearn.metrics
from sklearn.metrics import make_scorer, recall_score
from imblearn.ensemble import BalancedBaggingClassifier, BalancedRandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from utils import createLabels
import xgboost as xgb
import pandas as pd
import numpy as np
from copulas.univariate import GaussianKDE
from copulas.bivariate import Clayton
from copulas.multivariate import GaussianMultivariate
from copulas.visualization import scatter_2d
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc



X = []
Y = []
VALID_SIZE = 0.15 / 0.85    # We aim to get a 15% of the data as validation test,
                            # onle 85% of the data is arriving to the train-validation data split,
                            # so we need to get 0.15/0.85 part of the data.  
TEST_SIZE = 0.15
ENABLE_KFOLD = True
RESULTS_PATH = ''
def objectiveSVC(trial, best_params = None):
    global VALID_SIZE
    global X
    global Y
    X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=VALID_SIZE)
    if best_params is None:
        svc_c = trial.suggest_float("svc_c", 1e-10, 1e10, log=True)
    else:
        global X_test
        global Y_test
        X_train, X_valid, y_train, y_valid = [X,X_test,Y,Y_test]
        svc_c = best_params['svc_c']
    classifier_obj = sklearn.svm.SVC(C=svc_c, gamma="auto")
    if ENABLE_KFOLD and best_params is None:
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        recall_scorer = make_scorer(recall_score)
        score = cross_val_score(classifier_obj, X, Y, cv=kfold, scoring=recall_scorer).mean()
        return score
    classifier_obj.fit(X_train, y_train)
    y_pred = classifier_obj.predict(X_valid)
    if best_params is not None:
            model_name = 'SVC'
            y_pred_prob = classifier_obj.predict_proba(X_valid)[:, 1]
            fpr, tpr, _ = roc_curve(y_valid, y_pred_prob)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{model_name} (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC)')
            plt.legend(loc="lower right")
            plt.savefig(join(RESULTS_PATH,f'{model_name}.png'))
    accuracy = sklearn.metrics.recall_score(y_valid, y_pred)
    return accuracy

def objectiveBalancedRandomForestClassifier(trial, best_params = None):
    global VALID_SIZE
    global X
    global Y
    X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=VALID_SIZE)
    if best_params is None:
        params = {
            'n_estimators': trial.suggest_categorical('n_estimators', list(range(50, 400, 50))),
            'max_depth': trial.suggest_int('max_depth', 2, 30),
            'sampling_strategy': trial.suggest_categorical('sampling_strategy', [0.5, 'not minority', 'all'])
        }
    else:
        global X_test
        global Y_test
        X_train, X_valid, y_train, y_valid = [X,X_test,Y,Y_test]
        params = best_params
    classifier_obj = BalancedRandomForestClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'],
                                                    replacement=True, sampling_strategy=params['sampling_strategy'])
    if ENABLE_KFOLD and best_params is None:
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        recall_scorer = make_scorer(recall_score)
        score = cross_val_score(classifier_obj, X, Y, cv=kfold, scoring=recall_scorer).mean()
        return score

    classifier_obj.fit(X_train, y_train)
    y_pred = classifier_obj.predict(X_valid)
    if best_params is not None:
            model_name = 'BalancedRandomForest'
            y_pred_prob = classifier_obj.predict_proba(X_valid)[:, 1]
            fpr, tpr, _ = roc_curve(y_valid, y_pred_prob)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{model_name} (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC)')
            plt.legend(loc="lower right")
            plt.savefig(join(RESULTS_PATH,f'{model_name}.png'))
    accuracy = sklearn.metrics.recall_score(y_valid, y_pred)
    return accuracy


def objectiveBalancedBaggingClassifier(trial, best_params = None):
    global VALID_SIZE
    global X
    global Y
    X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=VALID_SIZE)
    if best_params is None:
        params = {
            'n_estimators': trial.suggest_categorical('n_estimators', list(range(50, 400, 50))),
            'sampling_strategy': trial.suggest_categorical('sampling_strategy', [0.5, 'not minority', 'all'])
        }
    else:
        global X_test
        global Y_test
        X_train, X_valid, y_train, y_valid = [X,X_test,Y,Y_test]
        params = best_params
    classifier_obj = BalancedBaggingClassifier(replacement=True, sampling_strategy=params['sampling_strategy'],
                                               n_estimators=params['n_estimators'])
    if ENABLE_KFOLD and best_params is None:
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        recall_scorer = make_scorer(recall_score)
        score = cross_val_score(classifier_obj, X, Y, cv=kfold, scoring=recall_scorer).mean()
        return score
    classifier_obj.fit(X_train, y_train)
    y_pred = classifier_obj.predict(X_valid)
    if best_params is not None:
            model_name = 'BalancedBagging'
            y_pred_prob = classifier_obj.predict_proba(X_valid)[:, 1]
            fpr, tpr, _ = roc_curve(y_valid, y_pred_prob)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{model_name} (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC)')
            plt.legend(loc="lower right")
            plt.savefig(join(RESULTS_PATH,f'{model_name}.png'))
    accuracy = sklearn.metrics.recall_score(y_valid, y_pred)
    return accuracy


def objectiveRandomForest(trial, best_params = None):
    global VALID_SIZE
    global X
    global Y
    X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=VALID_SIZE)
    if best_params is None:
        params = {
            'n_estimators': trial.suggest_categorical('n_estimators', list(range(50, 400, 50))),
            'max_depth': trial.suggest_int('max_depth',2,30),
        }
    else:
        global X_test
        global Y_test
        X_train, X_valid, y_train, y_valid = [X,X_test,Y,Y_test]
        params = best_params
    classifier_obj = sklearn.ensemble.RandomForestClassifier(max_depth=params['max_depth'],n_estimators=params['n_estimators'])
    if ENABLE_KFOLD and best_params is None:
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        recall_scorer = make_scorer(recall_score)
        score = cross_val_score(classifier_obj, X, Y, cv=kfold, scoring=recall_scorer).mean()
        return score
    classifier_obj.fit(X_train, y_train)
    y_pred = classifier_obj.predict(X_valid)
    if best_params is not None:
            model_name = 'RandomForest'
            y_pred_prob = classifier_obj.predict_proba(X_valid)[:, 1]
            fpr, tpr, _ = roc_curve(y_valid, y_pred_prob)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{model_name} (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC)')
            plt.legend(loc="lower right")
            plt.savefig(join(RESULTS_PATH,f'{model_name}.png'))
    accuracy = sklearn.metrics.recall_score(y_valid, y_pred)
    return accuracy


def objectiveXgboost(trial, best_params = None):
    global VALID_SIZE
    global X
    global Y
    X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=VALID_SIZE)

    if best_params is None:
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
        
    else:
        global X_test
        global Y_test
        X_train, X_valid, y_train, y_valid = [X,X_test,Y,Y_test]
        param = {
            "verbosity": 0,
            "objective": "binary:logistic",
            # use exact for small dataset.
            "tree_method": "exact",
            # defines booster, gblinear for linear functions.
            "booster": best_params["booster"],
            # L2 regularization weight.
            "lambda": best_params["lambda"],
            # L1 regularization weight.
            "alpha": best_params["alpha"],
            # sampling ratio for training data.
            "subsample": best_params["subsample"],
            # sampling according to each tree.
            "colsample_bytree": best_params["colsample_bytree"],
        }

        if param["booster"] in ["gbtree", "dart"]:
            # maximum depth of the tree, signifies complexity of the tree.
            param["max_depth"] = best_params["max_depth"]
            # minimum child weight, larger the term more conservative the tree.
            param["min_child_weight"] = best_params["min_child_weight"]
            param["eta"] = best_params["eta"]
            # defines how selective algorithm is.
            param["gamma"] = best_params["gamma"]
            param["grow_policy"] = best_params["grow_policy"]

        if param["booster"] == "dart":
            param["sample_type"] = best_params["sample_type"]
            param["normalize_type"] =best_params["normalize_type"]
            param["rate_drop"] = best_params["rate_drop"]
            param["skip_drop"] = best_params["skip_drop"]
            
    if ENABLE_KFOLD and best_params is None:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        accuracies = []
        for train_index, valid_index in skf.split(X, Y):
            X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
            y_train, y_valid = Y.iloc[train_index], Y.iloc[valid_index]

            dtrain = xgb.DMatrix(X_train, label=y_train)
            dvalid = xgb.DMatrix(X_valid, label=y_valid)

            model = xgb.train(param, dtrain)
            
            # Use accuracy for evaluation
            preds = model.predict(dvalid)
            pred_labels = np.rint(preds)  # Round to 0 or 1
            accuracy = sklearn.metrics.recall_score(y_valid, pred_labels)
            accuracies.append(accuracy)
        return np.mean(accuracies)
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)
    bst = xgb.train(param, dtrain)
    preds = bst.predict(dvalid)
    pred_labels = np.rint(preds)
    accuracy = sklearn.metrics.recall_score(y_valid, pred_labels)
    return accuracy


def objectiveLGBM(trial, best_params = None):
    global VALID_SIZE
    global X
    global Y
    X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=VALID_SIZE)

    if best_params is None:
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
    else:
        global X_test
        global Y_test
        X_train, X_valid, y_train, y_valid = [X,X_test,Y,Y_test]
        param = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "lambda_l1": best_params["lambda_l1"],
            "lambda_l2": best_params["lambda_l2"],
            "num_leaves": best_params["num_leaves"],
            "feature_fraction": best_params["feature_fraction"],
            "bagging_fraction": best_params["bagging_fraction"],
            "bagging_freq": best_params["bagging_freq"],
            "min_child_samples": best_params["min_child_samples"],
        }
    if ENABLE_KFOLD and best_params is None:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        accuracies = []
        for train_index, valid_index in skf.split(X, Y):
            X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
            y_train, y_valid = Y.iloc[train_index], Y.iloc[valid_index]

            dtrain = lgb.Dataset(X_train, label=y_train)
            model = lgb.train(param, dtrain)
            
            # Use accuracy for evaluation
            preds = model.predict(X_valid)
            pred_labels = np.rint(preds)  # Round to 0 or 1
            accuracy = sklearn.metrics.recall_score(y_valid, pred_labels)
            accuracies.append(accuracy)
        return np.mean(accuracies)
    
    dtrain = lgb.Dataset(X_train, label=y_train)
    gbm = lgb.train(param, dtrain)
    preds = gbm.predict(X_valid)
    pred_labels = np.rint(preds)
    accuracy = sklearn.metrics.recall_score(y_valid, pred_labels)

    return accuracy


if __name__ == "__main__":
    DATA_PATH =  r'C:\Users\Nitsan Cooper\OneDrive\מסמכים\DiabetesMLProject\data\diabetic_data.csv'
    SAVE_PATH =  r'C:\Users\Nitsan Cooper\OneDrive\מסמכים\DiabetesMLProject\results'
    data = createLabels(DATA_PATH)
    print('start data handling')
    data_encoded = data.copy()
    X_all = data_encoded.drop(['readmitted', 'readmitted_less_than_30', 'encounter_id', 'patient_nbr'], axis=1)
    y_all = data_encoded['readmitted_less_than_30'].astype(bool)
    # positive_samples = X_all[y_all == True]
    # negative_samples = X_all[y_all == False]
    
    # copula_model_positive = GaussianMultivariate()
    # copula_model_negative = GaussianMultivariate()
    # copula_model_positive.fit(positive_samples)
    # copula_model_negative.fit(negative_samples)
    # num_samples_per_class = 10000
    # synthetic_positive_samples = copula_model_positive.sample(num_samples_per_class)
    # synthetic_negative_samples = copula_model_negative.sample(num_samples_per_class)
    # synthetic_data_x = pd.concat([synthetic_positive_samples, synthetic_negative_samples], ignore_index=True)
    # synthetic_data_y = pd.Series([True] * num_samples_per_class + [False] * num_samples_per_class)

    X, X_test, Y, Y_test = train_test_split(X_all, y_all, test_size=VALID_SIZE, stratify=y_all)
    functions = [[objectiveBalancedRandomForestClassifier, "BalancedRandomForestClassifier"],
                [objectiveLGBM, 'lgbm'],
                [objectiveXgboost, 'xgboost'],
                [objectiveRandomForest, 'RandomForest'],
                [objectiveBalancedBaggingClassifier, "BalancedBaggingClassifier"]]
    result_path = SAVE_PATH
    if not os.path.isdir(result_path):
        os.mkdir(result_path)
    result_path = os.path.join(result_path,datetime.datetime.now().strftime("%y%m%d%H%M%S"))
    RESULTS_PATH = result_path
    os.mkdir(result_path)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    for objective, study_name in functions:
        print(study_name)
        study = optuna.create_study(direction="maximize", study_name=study_name)
        study.optimize(objective, n_trials=2, n_jobs=-1, show_progress_bar=True)
        total_score = objective(trial = None , best_params = study.best_trial.params)
        with open(result_path + "/" + study_name, "w") as file:
            print("Number of finished trials for {}: {}".format(study_name,len(study.trials)))
            file.write("Number of finished trials: {}\n".format(len(study.trials)))
            print(f'Using KFOLD: {ENABLE_KFOLD}')
            file.write(f'Using KFOLD: {ENABLE_KFOLD}')
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
            print(f'Score on test: {total_score}')
            file.write(f'Score on test: {total_score}')



