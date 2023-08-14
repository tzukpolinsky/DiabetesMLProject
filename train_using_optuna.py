"""
Optuna example that optimizes a classifier configuration for cancer dataset using LightGBM.

In this example, we optimize the validation accuracy of cancer detection using LightGBM.
We optimize both the choice of booster model and their hyperparameters.

"""
import datetime
import os.path
import sys
import joblib

import matplotlib.pyplot
import numpy as np
import optuna

import lightgbm as lgb
import sklearn.datasets
import sklearn.metrics
from imblearn.ensemble import BalancedBaggingClassifier, BalancedRandomForestClassifier, RUSBoostClassifier, \
    EasyEnsembleClassifier
from matplotlib import pyplot as plt
from optuna.visualization import plot_optimization_history, plot_intermediate_values, plot_param_importances, \
    plot_slice, plot_contour, plot_parallel_coordinate
from sklearn.model_selection import train_test_split

from utils import createLabels
import xgboost as xgb
from sklearn.metrics import classification_report, balanced_accuracy_score, roc_curve, auc, f1_score, \
    average_precision_score, fbeta_score, zero_one_loss, recall_score, confusion_matrix
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

X = []
Y = []
TEST_SIZE = 0.25


def getMetrics(y_test, y_pred):
    # avg_precision_score = average_precision_score(y_test, y_pred)
    # re_score = recall_score(y_test, y_pred)
    # loss = zero_one_loss(y_test, y_pred, normalize=True)
    # return avg_precision_score, loss, re_score
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    return specificity, roc_auc


def objectiveSVC(trial):
    global TEST_SIZE
    global X
    global Y
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=TEST_SIZE, stratify=Y)

    svc_c = trial.suggest_float("svc_c", 1e-10, 1e10, log=True)
    classifier_obj = sklearn.svm.SVC(C=svc_c, gamma="auto")
    classifier_obj.fit(x_train, y_train)
    y_pred = classifier_obj.predict(x_test)
    return getMetrics(y_test, y_pred)


def objectiveGradientBoostingClassifier(trail):
    global TEST_SIZE
    global X
    global Y
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=TEST_SIZE, stratify=Y)
    params = {
        'n_estimators': trail.suggest_categorical('n_estimators', list(range(10, 400, 50))),
        'max_depth': trail.suggest_int('max_depth', 2, 30),
    }
    classifier_obj = GradientBoostingClassifier(n_estimators=params['n_estimators'],
                                                max_depth=params['max_depth'])
    classifier_obj.fit(x_train, y_train)
    y_pred = classifier_obj.predict(x_test)
    return getMetrics(y_test, y_pred)


def objectiveXGBoostPruning(trial):
    global TEST_SIZE
    global X
    global Y
    train_x, valid_x, train_y, valid_y = train_test_split(X, Y, test_size=TEST_SIZE, stratify=Y)
    dtrain = xgb.DMatrix(train_x, label=train_y)
    param = {
        "verbosity": 0,
        "objective": "binary:logistic",
        "eval_metric": "error",
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
    }

    if param["booster"] == "gbtree" or param["booster"] == "dart":
        param["max_depth"] = trial.suggest_int("max_depth", 1, 9)
        param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "valid-error")
    history = xgb.cv(param, dtrain, num_boost_round=100, callbacks=[pruning_callback])

    mean_auc = history["valid-error-mean"].values[-1]
    return mean_auc


def objectiveEasyEnsembleClassifier(trail):
    global TEST_SIZE
    global X
    global Y
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=TEST_SIZE, stratify=Y)
    params = {
        'n_estimators': trail.suggest_categorical('n_estimators', list(range(10, 50, 1))),
        'sampling_strategy': trail.suggest_float("sampling_strategy", 0.1, 1.0, log=False)
    }
    classifier_obj = EasyEnsembleClassifier(n_estimators=params['n_estimators'],
                                            replacement=True, sampling_strategy=params['sampling_strategy'])
    classifier_obj.fit(x_train, y_train)
    y_pred = classifier_obj.predict(x_test)

    return getMetrics(y_test, y_pred)


def objectiveRUSBoostClassifier(trail):
    global TEST_SIZE
    global X
    global Y
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=TEST_SIZE, stratify=Y)
    params = {
        'n_estimators': trail.suggest_categorical('n_estimators', list(range(10, 50, 1))),
        'sampling_strategy': trail.suggest_float("sampling_strategy", 0.1, 1.0, log=False),
        'learning_rate': trail.suggest_float("learning_rate", 1e-8, 1.0, log=True)
    }
    classifier_obj = RUSBoostClassifier(n_estimators=params['n_estimators'], learning_rate=params['learning_rate'],
                                        replacement=True, sampling_strategy=params['sampling_strategy'])
    classifier_obj.fit(x_train, y_train)
    y_pred = classifier_obj.predict(x_test)

    return getMetrics(y_test, y_pred)


def objectiveBalancedRandomForestClassifier(trail):
    global TEST_SIZE
    global X
    global Y
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=TEST_SIZE, stratify=Y)
    params = {
        'n_estimators': trail.suggest_categorical('n_estimators', list(range(10, 50, 1))),
        'max_depth': trail.suggest_int('max_depth', 2, 20),
        'sampling_strategy': trail.suggest_float("sampling_strategy", 0.1, 1.0, log=False)
    }
    classifier_obj = BalancedRandomForestClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'],
                                                    replacement=True, sampling_strategy=params['sampling_strategy'])
    classifier_obj.fit(x_train, y_train)
    y_pred = classifier_obj.predict(x_test)

    return getMetrics(y_test, y_pred)


def objectiveBalancedBaggingClassifier(trail):
    global TEST_SIZE
    global X
    global Y
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=TEST_SIZE, stratify=Y)
    params = {
        'n_estimators': trail.suggest_categorical('n_estimators', list(range(10, 400, 10))),
        'sampling_strategy': trail.suggest_categorical('sampling_strategy', [0.5, 'not minority', 'all'])
    }
    classifier_obj = BalancedBaggingClassifier(replacement=True, sampling_strategy=params['sampling_strategy'],
                                               n_estimators=params['n_estimators'])
    classifier_obj.fit(x_train, y_train)
    y_pred = classifier_obj.predict(x_test)

    # return getMetrics(y_test, y_pred)
    return getMetrics(y_test, y_pred)


def objectiveRandomForest(trial):
    global TEST_SIZE
    global X
    global Y
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=TEST_SIZE, stratify=Y)
    params = {
        'n_estimators': trial.suggest_categorical('n_estimators', list(range(50, 400, 10))),
        'max_depth': trial.suggest_int('max_depth', 2, 30),
    }
    classifier_obj = sklearn.ensemble.RandomForestClassifier(max_depth=params['max_depth'],
                                                             n_estimators=params['n_estimators'])
    classifier_obj.fit(x_train, y_train)
    y_pred = classifier_obj.predict(x_test)

    return getMetrics(y_test, y_pred)


def objectiveXgboost(trial):
    global TEST_SIZE
    global X
    global Y
    train_x, valid_x, train_y, valid_y = train_test_split(X, Y, test_size=TEST_SIZE, stratify=Y)
    dtrain = xgb.DMatrix(train_x, label=train_y)
    dvalid = xgb.DMatrix(valid_x, label=valid_y)

    param = {
        "verbosity": 0,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        # sampling ratio for training data.
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        # sampling according to each tree.
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
    }

    if param["booster"] == "gbtree" or param["booster"] == "dart":
        param["max_depth"] = trial.suggest_int("max_depth", 1, 9)
        # minimum child weight, larger the term more conservative the tree.
        param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
        param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)
    xgb_cv_results = xgb.cv(
        params=param,
        dtrain=dtrain,
        num_boost_round=10000,
        nfold=5,
        stratified=True,
        early_stopping_rounds=100,
        seed=42,
        verbose_eval=False,
    )

    # Set n_estimators as a trial attribute; Accessible via study.trials_dataframe().
    trial.set_user_attr("n_estimators", len(xgb_cv_results))
    # Extract the best score.
    best_score = xgb_cv_results["test-auc-mean"].values[-1]
    return best_score


def objectiveLGBM(trial):
    global TEST_SIZE
    global X
    global Y
    train_x, valid_x, train_y, valid_y = train_test_split(X, Y, test_size=TEST_SIZE, stratify=Y)
    dtrain = lgb.Dataset(train_x, label=train_y)

    param = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 512),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }

    gbm = lgb.train(param, dtrain)
    preds = gbm.predict(valid_x)
    y_pred = np.rint(preds)
    return getMetrics(y_test, y_pred)


def get_test_results(algo_name, params, x_train, x_test, y_train, y_test):
    y_pred = None
    if algo_name == "lgbm":
        dtrain = lgb.Dataset(x_train, label=y_train)
        classifier_obj = lgb.train(params, dtrain)
        preds = classifier_obj.predict(x_test)
        y_pred = np.rint(preds)
    elif "xgboost" in algo_name:
        dtrain = xgb.DMatrix(x_train, label=y_train)
        dvalid = xgb.DMatrix(x_test, label=y_test)
        classifier_obj = xgb.train(params, dtrain)

        # Use accuracy for evaluation
        preds = classifier_obj.predict(dvalid)
        y_pred = np.rint(preds)  # Round to 0 or 1
    elif algo_name == "RandomForest":
        classifier_obj = sklearn.ensemble.RandomForestClassifier(max_depth=params['max_depth'],
                                                                 n_estimators=params['n_estimators'])
        classifier_obj.fit(x_train, y_train)
        y_pred = classifier_obj.predict(x_test)

    elif algo_name == "BalancedBaggingClassifier":
        classifier_obj = BalancedBaggingClassifier(replacement=True, sampling_strategy=params['sampling_strategy'],
                                                   n_estimators=params['n_estimators'])
        classifier_obj.fit(x_train, y_train)
        y_pred = classifier_obj.predict(x_test)

    elif algo_name == "BalancedRandomForestClassifier":
        classifier_obj = BalancedRandomForestClassifier(n_estimators=params['n_estimators'],
                                                        max_depth=params['max_depth'],
                                                        replacement=True, sampling_strategy=params['sampling_strategy'])
        classifier_obj.fit(x_train, y_train)
        y_pred = classifier_obj.predict(x_test)

    elif algo_name == "GradientBoostingClassifier":
        classifier_obj = GradientBoostingClassifier(n_estimators=params['n_estimators'],
                                                    max_depth=params['max_depth'],
                                                    learning_rate=params['learning_rate'])
        classifier_obj.fit(x_train, y_train)
        y_pred = classifier_obj.predict(x_test)

    elif algo_name == "RUSBoostClassifier":
        classifier_obj = RUSBoostClassifier(n_estimators=params['n_estimators'], learning_rate=params['learning_rate'],
                                            replacement=True, sampling_strategy=params['sampling_strategy'])
        classifier_obj.fit(x_train, y_train)
        y_pred = classifier_obj.predict(x_test)
    elif algo_name == "EasyEnsembleClassifier":
        classifier_obj = EasyEnsembleClassifier(n_estimators=params['n_estimators'],
                                                replacement=True, sampling_strategy=params['sampling_strategy'])
        classifier_obj.fit(x_train, y_train)
        y_pred = classifier_obj.predict(x_test)
    else:
        print("no algo with that name " + algo_name)
        return None
    class_report = classification_report(y_test, y_pred)
    return class_report, classifier_obj


def save_results(trail, trail_name, study_name, amount_of_trails, trail_index, save_path):
    with open(save_path + "/" + trail_name + "_results.txt", "w") as file:
        print("Number of finished trials for {}: {}".format(study_name, amount_of_trails))
        file.write("Number of finished trials: {} \n".format(amount_of_trails))
        print("{} Best trial:".format(trail_name))
        file.write("{} Best trial:\n".format(trail_name))
        print("Value: {}".format(trail.values))
        file.write("Value: {}\n".format(trail.values))
        print("Params: ")
        file.write("Params: \n")
        for key, value in trail.params.items():
            print("    {}: {}".format(key, value))
            file.write("    {}: {}\n".format(key, value))
        test_results, classifier_obj = get_test_results(study_name, trail.params, X, x_test, Y, y_test)
        print("classification_report on test: ")
        print(str(test_results))
        file.write("accuracy on test:\n")
        file.write(str(test_results))
        y_pred_prob = classifier_obj.predict_proba(x_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        plt.clf()
        plt.plot(fpr, tpr, label=f'{study_name} (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.savefig(save_path + "/" + trail_name + "_roc_curve.png", dpi=300)

        fig = plot_optimization_history(study, target=lambda t: t.values[trail_index])
        fig.write_image(save_path + "plot_optimization_history.png")


if __name__ == "__main__":
    data = createLabels(sys.argv[1])
    data_encoded = data.copy()
    X_all = data_encoded.drop(['readmitted', 'readmitted_less_than_30', 'encounter_id', 'patient_nbr'], axis=1)
    y_all = data_encoded['readmitted_less_than_30'].astype(bool)
    X, x_test, Y, y_test = train_test_split(X_all, y_all, test_size=TEST_SIZE, stratify=y_all)

    functions = [
        [objectiveXgboost, "xgboost"],
        [objectiveGradientBoostingClassifier, "GradientBoostingClassifier"],
        [objectiveBalancedRandomForestClassifier, "BalancedRandomForestClassifier"],
        [objectiveRUSBoostClassifier, 'RUSBoostClassifier'],
        [objectiveEasyEnsembleClassifier, 'EasyEnsembleClassifier']]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    result_path = sys.argv[2]
    if not os.path.isdir(result_path):
        os.mkdir(result_path)
    result_path = os.path.join(result_path, datetime.datetime.now().strftime("%y%m%d%H%M%S"))
    os.mkdir(result_path)
    n_trials = int(sys.argv[3])
    for objective, study_name in functions:
        study = optuna.create_study(
            directions=[optuna.study.StudyDirection.MAXIMIZE, optuna.study.StudyDirection.MAXIMIZE],
            study_name=study_name)
        study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True)
        save_path = result_path + "/" + study_name
        os.mkdir(save_path)
        highest_specificity_trail = max(study.best_trials, key=lambda t: t.values[0])
        # lowest_los_trail = min(study.best_trials, key=lambda t: t.values[1])
        highest_roc_auc_trail = max(study.best_trials, key=lambda t: t.values[1])

        save_results(highest_specificity_trail, "highest specificity", study_name, n_trials, 0, save_path)
        # save_results(lowest_los_trail, "lowest loss", study_name, n_trials, 1, save_path)
        save_results(highest_roc_auc_trail, "highest roc_auc", study_name, n_trials, 2, save_path)
        joblib.dump(study, save_path + "/study.pkl")
