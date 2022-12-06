import pandas as pd
import numpy as np
import os 
import re
import dump
import argparse
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')
from dataloader import load_data_ml
from tqdm import tqdm

# auto-sklearn with automatic hyperparameter tuning 
# with Bayesian Optimization based approach
from autosklearn.classification import AutoSklearnClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from skopt import BayesSearchCV 
import skopt

# conventional machine learning methods
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
# neural network method 
from sklearn.neural_network import MLPClassifier

# evaluation metrics
from sklearn.metrics import accuracy_score


def get_ml_model(model_name):
    if model_name.lower() == 'svm':
        # initialize optimizer
        # multi-class classification
        clf = svm.SVC(cache_size=2000, probability=True) # decision_function_shape='ovr')
        # search space for SVC
        params = dict()
        params['C'] = (1e-6, 100.0, 'log-uniform')
        params['gamma'] = (1e-6, 100.0, 'log-uniform')
        params['degree'] = (1,5)
        params['kernel'] = ['linear', 'poly', 'rbf', 'sigmoid']
    elif model_name.lower() == 'naive_bayes':
        clf = GaussianNB()
        # search space for GaussianNB
        params = dict()
        params['var_smoothing'] = (1e-9, 100.0, 'log-uniform')
    elif model_name.lower() == 'decision_trees':
        # initialize optimizer
        clf = DecisionTreeClassifier(random_state=0)
        # search space for GaussianNB
        params = dict()
        params['criterion'] = ['gini', 'entropy', ]#'log_loss']
        params['max_depth'] = (1,50)
        params['min_samples_split'] = (2,10)
        params['min_samples_leaf'] = (1,10)
    elif model_name.lower() == 'mlp':
        # initialize optimizer
        clf = MLPClassifier(random_state=1)
        # search space for MLP
        params = dict()
        params['activation'] = ['identity', 'logistic', 'tanh', 'relu']
        params['solver'] = ['lbfgs', 'sgd', 'adam']
        params['alpha'] = (1e-5, 100.0, 'log-uniform')
        params['learning_rate'] = ['constant', 'invscaling', 'adaptive']
        params['learning_rate_init'] = [0.1, 0.01, 0.001, 0.0001]
        params['max_iter'] = [200, 300, 400, 500]
    else:
        raise ValueError
    
    return clf, params


if __name__ == "__main__":
    save_path = './saved'
    if not os.path.exists(save_path): os.makedirs(save_path)
    params_path = './best_params'
    if not os.path.exists(params_path): os.makedirs(params_path)
    for area in (0, 1, 2, 3, 4, 5, 6, 7):
        print(f"==>Area_{area}: ")
        train_data, train_target = load_data_ml(area, 'train')
        valid_data, valid_target = load_data_ml(area, 'valid')
        print("Train Data Shape: ", train_data.shape, train_target.shape)
        print("Val Data Shape: ", valid_data.shape, valid_target.shape)
        with open(os.path.join(params_path, f"{area}_best_params.txt"), 'w') as f:
            for model_name in ['svm', 'naive_bayes', 'decision_trees', 'mlp']:
                print(f"==>{model_name}_Model: ")
                # get model
                clf, params = get_ml_model(model_name)

                # cross validation method
                cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
                # hyper-parameter searching
                # https://scikit-optimize.github.io/stable/modules/generated/skopt.BayesSearchCV.html
                search = BayesSearchCV(estimator=clf, search_spaces=params, n_jobs=6, cv=cv, n_iter=10,)# scoring ='neg_log_loss')
                # training and prediction
                search.fit(train_data, train_target)
                skopt.dump(search, os.path.join(save_path, f'{area}_{model_name}_model.joblib')) # save best model
                # search = skopt.load(f'{area}_{model_name}_model.joblib')
                # score on best result
                print("Best Scores: ", search.best_score_)
                print("Best Params: ", search.best_params_)
                acc_train = round(search.score(train_data, train_target) * 100, 3)
                print(f"Training Accuracy: {acc_train} %")
                if valid_data.shape[0] != 0:
                    pred_target = search.predict(valid_data)
                    # print(valid_target, pred_target)
                    acc_val = accuracy_score(valid_target, pred_target)
                    print(f"Validation Accuracy: {acc_val} %")
                else:
                    acc_val = None
                f.write(model_name + '\t' + str(search.best_score_) + '\t' + str(search.best_params_) + \
                        '\t' + str(acc_train) + '\t' + str(acc_val) + '\n')