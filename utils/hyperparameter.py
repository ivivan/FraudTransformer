VERSION = '1.0'

from hyperopt import hp
import numpy as np

import optuna

def set_hyperparameter():
    params = {
        "boosting_type": "gbdt",
        "objective": "binary",
        "metric": ['binary_logloss', 'auc'],
        "verbose": -1,
        "boost_from_average": True,
        "feature_pre_filter": False
    }

    max_depth_candid = [3, 4, 5, 6, 7]
    num_leaves_candid = np.linspace(5, 80, 16, dtype=int)
    bagging_freq_candid = np.linspace(0, 50, 10, dtype=int)
    # max_bin_candid = [2**i-1 for i in np.linspace(5, 7, 3,dtype=int)]
    min_data_in_leaf_candid = np.linspace(200, 1000, 3, dtype=int)

    base_gridparams = {
        "learning_rate": hp.loguniform("learning_rate", np.log(0.03), np.log(0.2)),
        "max_depth": hp.choice("max_depth", max_depth_candid),
        "num_leaves": hp.choice("num_leaves", num_leaves_candid),
        "feature_fraction": hp.quniform("feature_fraction", 0.5, 1.0, 0.1),
        "bagging_fraction": hp.quniform("bagging_fraction", 0.5, 1.0, 0.1),
        "bagging_freq": hp.choice("bagging_freq", bagging_freq_candid),
        "reg_alpha": hp.uniform("reg_alpha", 0, 0.01),
        "reg_lambda": hp.uniform("reg_lambda", 0, 0.01),
        # "max_bin": hp.choice("max_bin",max_bin_candid),
        "min_data_in_leaf": hp.choice("min_data_in_leaf",min_data_in_leaf_candid)
    }

    # return params,base_gridparams,max_depth_candid,num_leaves_candid,bagging_freq_candid,max_bin_candid,min_data_in_leaf_candid
    return params,base_gridparams,max_depth_candid,num_leaves_candid,bagging_freq_candid,min_data_in_leaf_candid


def set_hyperparameter_topuna():
    params = {
        'objective': 'binary',
        'metric': ['binary_logloss', 'auc'],
        'verbosity': -1,
        'boosting_type': 'gbdt',
        "boost_from_average": True,
        "feature_pre_filter": False,
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.01),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 100, 1000, step=100),
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
        'num_leaves': trial.suggest_int('num_leaves', 10, 100),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.1, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.1, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 0, 10)
    }

    # return params,base_gridparams,max_depth_candid,num_leaves_candid,bagging_freq_candid,max_bin_candid,min_data_in_leaf_candid
    return params




def main():
    params,base_gridparams,max_depth_candid,num_leaves_candid,bagging_freq_candid,min_data_in_leaf_candid = set_hyperparameter()
    print(params,base_gridparams,max_depth_candid,num_leaves_candid,bagging_freq_candid,min_data_in_leaf_candid)

if __name__ == "__main__":
    main()