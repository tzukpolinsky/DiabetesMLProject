"""
Optuna example that optimizes a classifier configuration for cancer dataset using LightGBM.

In this example, we optimize the validation accuracy of cancer detection using LightGBM.
We optimize both the choice of booster model and their hyperparameters.

"""
import copy
import datetime
import math
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
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler

from matplotlib import pyplot as plt
from optuna.visualization import plot_optimization_history, plot_intermediate_values, plot_param_importances, \
    plot_slice, plot_contour, plot_parallel_coordinate
from sklearn.model_selection import train_test_split

from utils import createLabels, clean_overlapping_data
import xgboost as xgb
from sklearn.metrics import classification_report, balanced_accuracy_score, roc_curve, auc, f1_score, \
    average_precision_score, fbeta_score, zero_one_loss, recall_score, confusion_matrix, precision_score, accuracy_score
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier

X = []
Y = []
TEST_SIZE = 0.25
TEST_SIZE_TRAIN = 0.75


def get_train_and_test():
    x_train, x_tst, y_train, y_tst = train_test_split(X, Y, test_size=TEST_SIZE_TRAIN)
    return x_train, x_tst, y_train, y_tst


def getMetrics(y_tst, y_pred):
    # avg_precision_score = average_precision_score(y_test, y_pred)
    # re_score = recall_score(y_test, y_pred)
    # loss = zero_one_loss(y_test, y_pred, normalize=True)
    # return avg_precision_score, loss, re_score
    # tn, fp, fn, tp = confusion_matrix(y_tst, y_pred).ravel()
    # specificity = tp / (tp + fn)
    # p = tn / (tn + fp * 1.5)
    recall = fbeta_score(y_tst, y_pred, beta=2)
    bas = balanced_accuracy_score(y_tst, y_pred)
    # f1 = precision_score(y_tst, y_pred)
    return bas, recall


def objectiveSVC(trial):
    global TEST_SIZE
    global X
    global Y
    x_train, x_test, y_train, y_test = get_train_and_test()

    svc_c = trial.suggest_float("svc_c", 1e-10, 1e10, log=True)
    classifier_obj = sklearn.svm.SVC(C=svc_c, gamma="auto")
    classifier_obj.fit(x_train, y_train)
    y_pred = classifier_obj.predict(x_test)
    return getMetrics(y_test, y_pred)


def objectiveGradientBoostingClassifier(trail):
    global TEST_SIZE
    global X
    global Y
    x_train, x_tst, y_train, y_tst = get_train_and_test()
    params = {
        'n_estimators': trail.suggest_categorical('n_estimators', list(range(10, 250, 10))),
        'max_depth': trail.suggest_int('max_depth', 2, 30),
        'min_samples_split': trail.suggest_int('min_samples_split', 2, 30),
        'learning_rate': 0.05,
        'subsample': trail.suggest_float("subsample", 0.1, 1.0, log=False)

    }
    classifier_obj = GradientBoostingClassifier(**params)
    classifier_obj.fit(x_train, y_train)
    y_pred = classifier_obj.predict(x_tst)
    return getMetrics(y_tst, y_pred)


def objectiveEasyEnsembleClassifier(trail):
    global TEST_SIZE
    global X
    global Y
    x_train, x_tst, y_train, y_tst = get_train_and_test()
    params = {
        'n_estimators': trail.suggest_categorical('n_estimators', list(range(10, 250, 10))),
        'sampling_strategy': 'all'
        # 'sampling_strategy': trail.suggest_float("sampling_strategy", 0.5, 1.0, log=False)
    }
    classifier_obj = EasyEnsembleClassifier(**params, replacement=True)
    classifier_obj.fit(x_train, y_train)
    y_pred = classifier_obj.predict(x_tst)

    return getMetrics(y_tst, y_pred)


def objectiveRUSBoostClassifier(trail):
    global TEST_SIZE
    global X
    global Y
    x_train, x_tst, y_train, y_tst = get_train_and_test()
    params = {
        'n_estimators': trail.suggest_categorical('n_estimators', list(range(10, 250, 10))),
        'sampling_strategy': 'all',
        # 'sampling_strategy': trail.suggest_float("sampling_strategy", 0.5, 1.0, log=False)
        'learning_rate': trail.suggest_float("learning_rate", 1e-8, 1.0, log=True)
    }
    classifier_obj = RUSBoostClassifier(**params, replacement=True)
    classifier_obj.fit(x_train, y_train)
    y_pred = classifier_obj.predict(x_tst)

    return getMetrics(y_tst, y_pred)


def objectiveBalancedRandomForestClassifier(trail):
    global TEST_SIZE
    global X
    global Y
    x_train, x_tst, y_train, y_tst = get_train_and_test()
    params = {
        'n_estimators': trail.suggest_categorical('n_estimators', list(range(10, 250, 10))),
        'max_depth': trail.suggest_int('max_depth', 10, 20),
        'min_samples_split': trail.suggest_int('min_samples_split', 2, 50),
        'sampling_strategy': 'all'
        # 'sampling_strategy': trail.suggest_float("sampling_strategy", 0.5, 1.0, log=False)
    }
    classifier_obj = BalancedRandomForestClassifier(**params,
                                                    replacement=True)
    classifier_obj.fit(x_train, y_train)
    y_pred = classifier_obj.predict(x_tst)

    return getMetrics(y_tst, y_pred)


def objectiveBalancedBaggingClassifier(trail):
    global TEST_SIZE
    global X
    global Y
    x_train, x_tst, y_train, y_tst = get_train_and_test()
    params = {
        'n_estimators': trail.suggest_categorical('n_estimators', list(range(10, 50, 1))),
        'sampling_strategy': 'all'
        # 'sampling_strategy': trail.suggest_float("sampling_strategy", 0.5, 1.0, log=False)
    }
    classifier_obj = BalancedBaggingClassifier(replacement=True, **params)
    classifier_obj.fit(x_train, y_train)
    y_pred = classifier_obj.predict(x_tst)

    # return getMetrics(y_test, y_pred)
    return getMetrics(y_tst, y_pred)


def objectiveRandomForest(trial):
    global TEST_SIZE
    global X
    global Y
    x_train, x_test, y_train, y_test = get_train_and_test()
    params = {
        'n_estimators': trial.suggest_categorical('n_estimators', list(range(10, 400, 10))),
        'max_depth': trial.suggest_int('max_depth', 2, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 30),
    }
    classifier_obj = RandomForestClassifier(**params)
    classifier_obj.fit(x_train, y_train)
    y_pred = classifier_obj.predict(x_test)

    return getMetrics(y_test, y_pred)


def objectiveXgboost(trial):
    global TEST_SIZE
    global X
    global Y
    x_train, x_tst, y_train, y_tst = get_train_and_test()
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dvalid = xgb.DMatrix(x_tst, label=y_tst)

    param = {
        "verbosity": 0,
        "objective": "binary:logistic",
        # use exact for small dataset.
        "tree_method": "gpu_hist",
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
        param["max_depth"] = trial.suggest_int("max_depth", 3, 19, step=2)
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
    return getMetrics(y_tst, pred_labels)


def objectiveLGBM(trial):
    global TEST_SIZE
    global X
    global Y
    train_x, valid_x, train_y, valid_y = get_train_and_test()
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
    elif algo_name == "RandomForestClassifier":
        classifier_obj = RandomForestClassifier(**params)
        classifier_obj.fit(x_train, y_train)
        y_pred = classifier_obj.predict(x_test)

    elif algo_name == "BalancedBaggingClassifier":
        classifier_obj = BalancedBaggingClassifier(**params)
        classifier_obj.fit(x_train, y_train)
        y_pred = classifier_obj.predict(x_test)

    elif algo_name == "BalancedRandomForestClassifier":
        classifier_obj = BalancedRandomForestClassifier(**params)
        classifier_obj.fit(x_train, y_train)
        y_pred = classifier_obj.predict(x_test)

    elif algo_name == "GradientBoostingClassifier":
        classifier_obj = GradientBoostingClassifier(**params)
        classifier_obj.fit(x_train, y_train)
        y_pred = classifier_obj.predict(x_test)

    elif algo_name == "RUSBoostClassifier":
        classifier_obj = RUSBoostClassifier(**params)
        classifier_obj.fit(x_train, y_train)
        y_pred = classifier_obj.predict(x_test)
    elif algo_name == "EasyEnsembleClassifier":
        classifier_obj = EasyEnsembleClassifier(**params)
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
        if "xgboost" in study_name:
            dvalid = xgb.DMatrix(x_test, label=y_test)
            y_pred_prob = classifier_obj.predict(dvalid, output_margin=False)
        else:
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
    #cleaned_data = clean_overlapping_data(data)

    X_all = data[:, :-1]
    y_all = data[:, -1]
    print("amount of pos before split to test and training: " + str(y_all.sum()))
    print("amount of neg before split to test and training: " + str(y_all.shape[0] - y_all.sum()))
    X, x_test, Y, y_test = train_test_split(X_all, y_all, test_size=TEST_SIZE,stratify=y_all)
    data_to_clean = []
    for i in range(Y.shape[0]):
        data_to_clean.append(copy.deepcopy(np.append(X[i],Y[i])))
    # sm = SMOTE(n_jobs=-1)
    # sm = SMOTETomek(random_state=42)
    # sm = ADASYN(n_neighbors=math.ceil(np.sum(Y) * 0.005))

    # sm = RandomUnderSampler(sampling_strategy='majority')
    cleaned_data = clean_overlapping_data(np.array(data_to_clean))
    X, Y = cleaned_data[:, :-1], cleaned_data[:, -1]  # sm.fit_resample(X, Y)
    print("amount of pos after split to test and training: " + str(Y.sum()))
    print("amount of neg after split to test and training: " + str(Y.shape[0] - Y.sum()))
    functions = [
        # [objectiveXgboost, "xgboost"],
        # [objectiveRandomForest, "RandomForestClassifier"],
        [objectiveBalancedRandomForestClassifier, "BalancedRandomForestClassifier"],
        # [objectiveBalancedBaggingClassifier, "BalancedBaggingClassifier"],
        [objectiveRUSBoostClassifier, 'RUSBoostClassifier'],
        [objectiveEasyEnsembleClassifier, 'EasyEnsembleClassifier'],
        [objectiveGradientBoostingClassifier, "GradientBoostingClassifier"]
    ]

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
        study.optimize(objective, n_trials=n_trials, n_jobs=-1, show_progress_bar=True)
        save_path = result_path + "/" + study_name
        os.mkdir(save_path)
        highest_specificity_trail = max(study.best_trials, key=lambda t: t.values[0])
        # lowest_los_trail = min(study.best_trials, key=lambda t: t.values[1])
        highest_roc_auc_trail = max(study.best_trials, key=lambda t: t.values[1])
        # lowest_false_positive_trail = max(study.best_trials, key=lambda t: t.values[2])
        # highest_true_positive_trail = max(study.best_trials, key=lambda t: t.values[3])

        save_results(highest_specificity_trail, "highest specificity", study_name, n_trials, 0, save_path)
        # save_results(lowest_los_trail, "lowest loss", study_name, n_trials, 1, save_path)
        save_results(highest_roc_auc_trail, "highest p ", study_name, n_trials, 1, save_path)
        # save_results(lowest_false_positive_trail, "lowest false positive", study_name, n_trials, 2, save_path)
        # save_results(highest_true_positive_trail, "highest true positive", study_name, n_trials, 3, save_path)
        joblib.dump(study, save_path + "/study.pkl")
