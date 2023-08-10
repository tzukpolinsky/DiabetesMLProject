import sys

import numpy as np
import lightgbm as lgb 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, auc, roc_curve, roc_auc_score
import optuna 
import pandas as pd
import os
import tarfile
import urllib.request
from sklearn.datasets import make_regression
import xgboost as xgb

import utils
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial, dtrain, n_estimators, params_in_stages): # categorical_feats
                  
        params = {  # Already tuned in stages but can try also with Optuna together with the rest of the hyperparameters 
                    # "learning_rate": trial.suggest_float("learning_rate", 0.01, 1),  
                    #"max_depth": trial.suggest_int("max_depth", 1, 12), 
                    # "gamma" : trial.suggest_float("gamma", 0, 5),
                    #"subsample": trial.suggest_float("subsample", 0.1, 1.0), 
                    
                    # Have yet to be tuned in stages 
                    "min_child_weight" : trial.suggest_int("min_child_weight", 1, 50), 
                    "lambda" : trial.suggest_float("lambda", 0.0, 0.3),
                    "alpha": trial.suggest_float("alpha", 0.0, 0.3), 
                    
                    # "scale_pos_weight" :trial.suggest_float("scale_pos_weight", 0.0, 1.0),  # for classification 
           }
        
        params.update(params_in_stages)

        
        xgb_cv = xgb.cv(params=params, num_boost_round = n_estimators, dtrain=dtrain,  nfold=10, shuffle=True,  stratified=False, metrics="rmse", seed=42) 
        
        return (xgb_cv['test-rmse-mean'].iloc[-1])




##  using default hyperparameter values 
# if __name__ == "__main__":
    

#         X = pd.read_csv('X.csv')
#         y = pd.read_csv('y.csv')


#         n_trials = 1000
        
#         early_stopping_rounds=5

#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, random_state=42)
        
#         X_train_es, X_val, y_train_es, y_val = train_test_split(X_train, y_train, shuffle=True, random_state=42)



#         xgb_reg = xgb.XGBRegressor(random_state=42)
#         xgb_reg.fit(X_train, y_train, verbose=False)
        
#         y_pred = xgb_reg.predict(X_val)
#         mse_preds = mean_squared_error(y_pred, y_val)
        
#        # Dictionary to save tuned hypereparameter values 
#         params = {}
        
#         xgb_reg = xgb.XGBRegressor(random_state=42, early_stopping_rounds=early_stopping_rounds)
#         xgb_reg.fit(X_train_es, y_train_es,
#                eval_set=[(X_val, y_val)], verbose=False)
        
#         y_pred = xgb_reg.predict(X_val)
#         mse_preds = mean_squared_error(y_pred, y_val)
#         print("rmse2 - to find number of estimators", np.sqrt(mse_preds))


#         num_estimators = xgb_reg.best_iteration
#         print('num_estimators', num_estimators)
#         params['num_estimators'] = num_estimators
        
#         xgb_reg = xgb.XGBRegressor(n_estimators=num_estimators, random_state=42)
#         xgb_reg.fit(X_train, y_train, verbose=False)

#         y_pred = xgb_reg.predict(X_val)
#         mse_preds = mean_squared_error(y_pred, y_val)
#         print("rmse3-val-to find number of estimators", np.sqrt(mse_preds))


       
#         max_depth = {'max_depth':range(3,20,1)}
#         gsearch = GridSearchCV(param_grid=max_depth, estimator=xgb.XGBRegressor(n_estimators=params['num_estimators'], seed=42), scoring='neg_mean_squared_error', cv=10)
        
#         gsearch.fit(X_train,y_train)
#         print(gsearch.best_params_) 
#         print('rmse3 - find max_depth', np.sqrt(np.negative(gsearch.best_score_)))
#         params.update(gsearch.best_params_)
        
#         lr_n_est= {'learning_rate':np.linspace(0.1,0.01, 10), "n_estimators":range(5,120,1)}  
#         gsearch= GridSearchCV(param_grid=lr_n_est, estimator=xgb.XGBRegressor(n_estimators=params['num_estimators'], max_depth=params['max_depth'], seed=42), scoring='neg_mean_squared_error', cv=10)

#         gsearch.fit(X_train,y_train)
#         print(gsearch.best_params_) 
#         print('rmse4 - learning rate and num estimators', np.sqrt(np.negative(gsearch.best_score_)))
#         params.update(gsearch.best_params_)
       
 
#         subsample = {'subsample':np.arange(0.8,1, 2)}  
#         gsearch = GridSearchCV(param_grid=subsample, estimator=xgb.XGBRegressor(n_estimators=params['num_estimators'], max_depth=params['max_depth'], learning_rate =gsearch.best_params_['learning_rate'] , seed=42), scoring='neg_mean_squared_error', cv=10)
#         gsearch.fit(X_train,y_train)
#         print(gsearch.best_params_) 
#         print('rmse', np.sqrt(np.negative(gsearch.best_score_)))
#         params.update(gsearch.best_params_)

     
#         gamma= {'gamma':np.arange(0, 5, 0.01)}  
#         gsearch = GridSearchCV(param_grid=gamma, estimator=xgb.XGBRegressor(n_estimators=params['num_estimators'], max_depth=params['max_depth'], learning_rate =params['learning_rate'], subsample=params['subsample'], seed=42), scoring='neg_mean_squared_error', cv=10)
#         gsearch.fit(X_train,y_train)
#         print(gsearch.best_params_) 
#         print('rmse', np.sqrt(np.negative(gsearch.best_score_)))
#         params.update(gsearch.best_params_)
       
#         params_in_stages = {'gamma':params['gamma'], 'subsample':params['subsample'], 'max_depth':params['max_depth'], 'learning_rate':params['learning_rate']}
        
#         # DMatrix is an internal data structure that is used by XGBoost, which is optimized for both memory efficiency and training speed. You can construct DMatrix from multiple different sources of data.
#         dtrain = xgb.DMatrix(X_train, y_train) 
        
#         study = optuna.create_study(direction='minimize')
#         func = lambda trial: objective(trial, dtrain, params['n_estimators'], params_in_stages) 
#         study.optimize(func, n_trials= n_trials) 
    
#         best_params = study.best_params
#         best_params.update(params_in_stages)
#         print("All Best params :", best_params)
#         best_model = xgb.XGBRegressor(**best_params, n_estimators=params['n_estimators'])
        
#         best_model.fit(X_train, y_train)
           
#        xgb_reg = xgb.XGBRegressor(random_state=42)
#        xgb_reg.fit(X_train, y_train, verbose=False)
#        y_pred = xgb_reg.predict(X_test)
#       mse_preds_default = mean_squared_error(y_pred, y_test)
#       print("rmse-test-default", np.sqrt(mse_preds_default))
     
#       print("We have reduced rmse by ", np.sqrt(mse_preds_default) - mean_squared_error(preds_test, y_test, squared=False))
       



## Using a set of commonly used start values
if __name__ == "__main__":
        data = utils.createLabels("C:\\Users\\tzuk9\\Documents\\dataset_diabetes\\diabetic_data.csv")
        data_encoded = data.copy()
        X = data_encoded.drop(['readmitted', 'readmitted_less_than_30', 'encounter_id', 'patient_nbr'], axis=1)
        y = data_encoded['readmitted_less_than_30'].astype(bool)
        X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, random_state=42)
        X_train_es, X_val, y_train_es, y_val = train_test_split(X_train, y_train, shuffle=True, random_state=42)
        ratio = float(np.sum(y_train_es == 1)) / np.sum(y_train_es==0)

        clf = xgb.XGBClassifier(
                                max_depth=12,
                                n_estimators=10,
                                learning_rate=0.1,
                                subsample=1.0,
                                colsample_bytree=0.5,
                                min_child_weight=3,
                                scale_pos_weight=ratio,
                                reg_alpha=0.03,
                                seed=1301)

        clf.fit(X_train, y_train, early_stopping_rounds=50, eval_metric="auc",
                eval_set=[(X_train, y_train), (X_test, y_test)])

        print('Overall AUC:', roc_auc_score(y, clf.predict_proba(X_val, ntree_limit=clf.best_iteration)[:, 1]))

        