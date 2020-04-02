# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 2018

@author: Alise Midtfjord
"""

#Defining fourteen different classifiers and seven different feature selectors.
"------------------Classification------------------"

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from pyearth import Earth
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from skrebate import ReliefF
from sklearn.feature_selection import RFE, SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA, FastICA 
from sklearn.model_selection import GridSearchCV
import warnings

def logrel1(X_std_train, y_train, state):
    '''
    Classification using Logistic regression with L1-regularization.
    
    :param str X_std_train: Training data 
    :param str y_train: Response to the training data
    :param int state: Random state
    :return: Classification model made from the training data
    '''
    
    param_range = [0.0001, 0.001,0.005, 0.01,0.05, 0.1, 1.0, 10.0, 100.0, 1000.0]
    param_grid = {'C' : param_range}
    gs = GridSearchCV(estimator=LogisticRegression(fit_intercept = True, random_state = state,
                                                   solver = 'liblinear',
                                                   penalty = 'l1',class_weight = 'balanced'),
                      param_grid=param_grid,
                      scoring='roc_auc',
                      cv=4)
  
    model = gs.fit(X_std_train,y_train)
    print('Training score Logistic regression:', gs.best_score_)
    print(gs.best_params_)
    return(model)


def logrel2(X_std_train, y_train, state):
    '''
    Classification using Logistic regression with L2-regularization.
    
    :param str X_std_train: Training data 
    :param str y_train: Response to the training data
    :param int state: Random state
    :return: Classification model made from the training data
    '''
    param_range = [0.0001, 0.001,0.005, 0.01,0.05, 0.1, 1.0, 10.0, 100.0, 1000.0]
    param_grid = {'C' : param_range}
    gs = GridSearchCV(estimator=LogisticRegression(fit_intercept = True, random_state = state,
                                                   solver = 'liblinear', class_weight = 'balanced',
                                                   penalty = 'l2'),
                      param_grid=param_grid,
                      scoring='roc_auc',
                      cv=4)
    
    model = gs.fit(X_std_train,y_train)
    print('Training score Logistic regression:', gs.best_score_)
    print(gs.best_params_)
    return(model)
    
def rf(X_train,y_train, X_test, y_test, state):
    '''
    Classification using Random Forests.
    
    :param str X_std_train: Training data 
    :param str y_train: Response to the training data
    :param int state: Random state
    :return: Classification model made from the training data
    '''
    param_range = [1, 2, 3, 4, 5, 6, 7, None]
    param_grid = {'max_depth': param_range}
    gs = GridSearchCV(estimator=RandomForestClassifier(
                                                   n_estimators=100,
                                                   n_jobs=-1, random_state = state, class_weight = 'balanced'),
                  param_grid=param_grid,
                  scoring='roc_auc',
                  cv=4)
    model = gs.fit(X_train,y_train)
    print('Training score Random forest:', model.best_score_)
    print(model.best_params_)
    return(model)


def knn(X_std_train, y_train):
    '''
    Classification using K-nearest-neighbors
    
    :param str X_std_train: Training data 
    :param str y_train: Response to the training data
    :return: Classification model made from the training data
    '''
    param_range = [2,5,8, 10, 15,20]
    feature_range = [10,30,50]
    param_grid = {'n_neighbors': param_range, 'leaf_size': feature_range}
    gs = GridSearchCV(estimator = KNeighborsClassifier(n_jobs = -1),
                      param_grid = param_grid,
                      scoring = 'roc_auc',
                      cv = 4)
    model = gs.fit(X_std_train,y_train)
    print('Training score KNN:', gs.best_score_)
    print(gs.best_params_)
    return(model)
    
def adaboostlog(X_std_train, y_train, state):
    '''
    Classification using Adaboost with Logistic Regression as base.
    
    :param str X_std_train: Training data 
    :param str y_train: Response to the training data
    :param int state: Random state
    :return: Classification model made from the training data
    '''
    base = LogisticRegression()
    param_range = [0.5,1, 2, 3]
    param_grid = {'learning_rate': param_range}
    gs = GridSearchCV(estimator = AdaBoostClassifier(base, n_estimators = 1000, 
                                                     random_state = state),
                      param_grid = param_grid,
                      scoring = 'roc_auc',
                      cv = 4)
    
    model = gs.fit(X_std_train,y_train)
    print('Training score AdaBoost:', gs.best_score_)
    print(gs.best_params_)
    return(model)
    
    
def decisiontree(X_train, y_train, state):
    '''
    Classification using Decision Trees.
    
    :param str X_std_train: Training data 
    :param str y_train: Response to the training data
    :param int state: Random state
    :return: Classification model made from the training data
    '''
    param_range = [1, 2, 3, 4, 5, 6, 7, None]
    param_grid = {'max_depth': param_range}
    
    
    gs = GridSearchCV(estimator = DecisionTreeClassifier(random_state = state, class_weight = 'balanced'),
                      param_grid = param_grid,
                      scoring = 'roc_auc',
                      cv = 4)
    
    model = gs.fit(X_train,y_train)
    print('Training score Decision Tree:', gs.best_score_)
    print(gs.best_params_)
    return(model)

    

def gnb(X_std_train, y_train, state):
    '''
    Classification using Gaussian Naives Bayes.
    
    :param str X_std_train: Training data 
    :param str y_train: Response to the training data
    :param int state: Random state
    :return: Classification model made from the training data
    '''
    param_grid = {}
    gs = GridSearchCV(estimator=GaussianNB(),
                      param_grid=param_grid,
                      scoring='roc_auc',
                      cv=4)
    
    model = gs.fit(X_std_train,y_train)
    print('Training score Linear GNB:', gs.best_score_)
    print(gs.best_params_)
    return(model)
    
def lda(X_std_train, y_train):
    '''
    Classification using LDA.
    
    :param str X_std_train: Training data 
    :param str y_train: Response to the training data
    :return: Classification model made from the training data
    '''
    param_range = ['auto',0.1,0.5,0.8,0]
    feature_range = [5,7,9,10]
    tol_range = [0.0001, 0.00001, 0.001, 0.01]
    param_grid = [{'solver' : ['lsqr'], 'shrinkage' : param_range, 'n_components' : feature_range,
                  'tol' : tol_range}, {'solver' : ['svd'], 'n_components' : feature_range,
                  'tol' : tol_range}]
    gs = GridSearchCV(estimator=LinearDiscriminantAnalysis(priors = [0.776,0.224]),
                      param_grid=param_grid,
                      scoring='roc_auc',
                      cv=4)
    
    warnings.filterwarnings('ignore')
    model = gs.fit(X_std_train,y_train)
    print('Training score Linear LDA:', gs.best_score_)
    print(gs.best_params_)
    warnings.filterwarnings('default')
    return(model)
    
def qda(X_std_train, y_train):
    '''
    Classification using QDA.
    
    :param str X_std_train: Training data 
    :param str y_train: Response to the training data
    :return: Classification model made from the training data
    '''
    tol_range = [0.0001, 0.00001, 0.001, 0.01]
    reg_range = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    param_grid = {'tol' : tol_range,'reg_param' : reg_range}
    gs = GridSearchCV(estimator=QuadraticDiscriminantAnalysis(priors = [0.776,0.224]),
                      param_grid=param_grid,
                      scoring='roc_auc',
                      cv=4)
    warnings.filterwarnings('ignore')
    model = gs.fit(X_std_train,y_train)
    print('Training score QDA:', gs.best_score_)
    print(gs.best_params_)
    warnings.filterwarnings('default')
    return(model)

def nnet(X_std_train, y_train, state):
    '''
    Classification using Neural Network.
    
    :param str X_std_train: Training data 
    :param str y_train: Response to the training data
    :param int state: Random state
    :return: Classification model made from the training data
    '''
    param_range = [5,10,50, 100, 150, 200]
    feature_range = [0.00001, 0.0001, 0.001, 0.01, 0.1]
    tol_range = [0.0001, 0.00001, 0.001, 0.01]
    param_grid = {'hidden_layer_sizes': param_range, 'alpha' : feature_range, 'tol' : tol_range}
    
    gs = GridSearchCV(estimator=MLPClassifier(random_state=state),
                      param_grid=param_grid,
                      scoring='roc_auc',
                      cv=4)
    warnings.filterwarnings('ignore')
    model = gs.fit(X_std_train,y_train)
    print('Training score Neural networkC:', gs.best_score_)
    print(gs.best_params_)
    warnings.filterwarnings('default')
    return(model)
    
def mars(X_std_train, y_train):
    '''
    Classification using Multivariate Adaptive Regression Splines.
    
    :param str X_std_train: Training data 
    :param str y_train: Response to the training data
    :return: Classification model made from the training data
    '''
    param_range = [1,3,5]
    feature_range = [0.01,0.05,0.1,0.2]
    param_grid = {'penalty' : param_range, 'minspan_alpha' : feature_range}
    gs = GridSearchCV(estimator=Earth(),
                     param_grid=param_grid,
                     scoring='roc_auc',
                     cv=4)
    
    warnings.filterwarnings('ignore')
    model = gs.fit(X_std_train,y_train)
    print('Training score MARS:', gs.best_score_)
    print(gs.best_params_)
    warnings.filterwarnings('default')
    return(model)


def plsr(X_std_train, y_train):
    '''
    Classification using Partial Least Squares.
    
    :param str X_std_train: Training data 
    :param str y_train: Response to the training data
    :return: Classification model made from the training data
    '''
    param_range =  np.arange(1,X_std_train.shape[1]+1)
    feature_range = [0.0000001, 0.000001, 0.00001]
    param_grid = {'n_components' : param_range, 'tol' : feature_range}
    gs = GridSearchCV(estimator=PLSRegression(scale = False),
                     param_grid=param_grid,
                     scoring='roc_auc',
                     cv=4)
    
    warnings.filterwarnings('ignore')
    model = gs.fit(X_std_train,y_train)
    print('Training score PLSR:', gs.best_score_)
    print(gs.best_params_)
    warnings.filterwarnings('default')
    return(model)
    
def svc(X_std_train, y_train, state):
    '''
    Classification using Support Vector Classifer.
    
    :param str X_std_train: Training data 
    :param str y_train: Response to the training data
    :param int state: Random state
    :return: Classification model made from the training data
    '''
    param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    feature_range = [0.001,0.05,0.1,0.2]
    cache_range = [50,100,200,300]
    param_grid = {'C': param_range, 'gamma' : feature_range, 'cache_size' : cache_range}
    gs = GridSearchCV(estimator=SVC(random_state = state, class_weight = 'balanced'),
                      param_grid=param_grid,
                      scoring='roc_auc',
                      cv=4)
    
    model = gs.fit(X_std_train,y_train)
    print('Training score SVC:', gs.best_score_)
    print(gs.best_params_)
    return(model)
    
def linearsvc(X_std_train, y_train, state):
    '''
    Classification using Linear Support Vector Classifier.
    
    :param str X_std_train: Training data 
    :param str y_train: Response to the training data
    :param int state: Random state
    :return: Classification model made from the training data
    '''
    param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    feature_range = [0.00001,0.0001,0.001,0.1,1]
    param_grid = {'C': param_range, 'tol' : feature_range}
    gs = GridSearchCV(estimator=LinearSVC(fit_intercept = True, random_state = state, class_weight = 'balanced'),
                      param_grid=param_grid,
                      scoring='roc_auc',
                      cv=4)
    
    model = gs.fit(X_std_train,y_train)
    print('Training score Linear SVC:', gs.best_score_)
    print(gs.best_params_)
    return(model)
    
"------------------Feature selection------------------"    
    
def relieff(X_std_train, X_std_test, y_train,n_features, colNames, features):
    '''
    Feature selection using ReliefF.
    
    :param str X_std_train: Training data 
    :param str X_std_test: Validation data
    :param str y_train: Response to the training data
    :param int n_features: Number of features to be selected
    :param colNames: List with the names of the columns/features
    :features: List that the selected features will be added to
    :return: The training data and validation data with only the selected features
             and the list with the features
    '''
    relieff = ReliefF(n_features_to_select=n_features, n_neighbors=20)
    relieff.fit(X_std_train,y_train)
    importances = relieff.feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_names = []
    
    for f in range(X_std_train.shape[1]):
        feature_names.append(colNames[indices[f]])
    print(feature_names[0:n_features])
    X_std_train = X_std_train[:,indices[0:n_features]]
    X_std_test = X_std_test[:,indices[0:n_features]]
    features.append(feature_names[0:n_features])
    return (X_std_train, X_std_test, features)

def rf_selection(X_train, X_test, y_train, colNames, state, n_features, features):
    '''
    Feature selection using Random Forests.
    
    :param str X_std_train: Training data 
    :param str X_std_test: Validation data
    :param str y_train: Response to the training data
    :param colNames: List with the names of the columns/features
    :param state: Random state
    :param int n_features: Number of features to be selected
    :features: List that the selected features will be added to
    :return: The training data and validation data with only the selected features
             and the list with the features
    '''
    forest = RandomForestClassifier(n_estimators=1000,
                                n_jobs=-1, class_weight = 'balanced', random_state=state)
    forest.fit(X_train, y_train)
    importances = forest.feature_importances_

    indices = np.argsort(importances)[::-1]
    feature_names = []
    
    for f in range(X_train.shape[1]):
        feature_names.append(colNames[indices[f]])
    print(feature_names[0:n_features])
    features.append(feature_names[0:n_features])

    X_train = X_train[:,indices[0:n_features]]
    X_test = X_test[:,indices[0:n_features]]
    return (X_train, X_test, features)

def logrel1_selection(X_std_train, X_std_test, y_train,n_features, state, colNames):
    '''
    Feature selection using L1-logistic regression.
    
    :param str X_std_train: Training data 
    :param str X_std_test: Validation data
    :param str y_train: Response to the training data
    :param int n_features: Number of features to be selected
    :param state: Random state
    :param colNames: List with the names of the columns/features
    :return: The training data and validation data with only the selected features
    '''
    leng = X_std_train.shape[1]
    logistic = LogisticRegression(penalty = 'l1',class_weight = 'balanced', random_state = state)
    logre = RFE(logistic, n_features,step = 10)
    X_std_train = logre.fit_transform(X_std_train, y_train)
    X_std_test = logre.transform(X_std_test)
    for i in range(leng):
        if logre.get_support()[i] == True:
            print (colNames[i])
    return(X_std_train, X_std_test)
    
def lda_selection(X_std_train, X_std_test, y_train):
    '''
    Dimension reduction using LDA.
    
    :param str X_std_train: Training data 
    :param str X_std_test: Validation data
    :param str y_train: Response to the training data
    :return: The training data and validation data with only the new component
    '''
    mod = LinearDiscriminantAnalysis(tol = 0.1)
    X_std_train = mod.fit_transform(X_std_train,y_train)
    X_std_test = mod.transform(X_std_test)
        
    return(X_std_train, X_std_test, mod) 
    
def pca(X_std_train, X_std_test, n_features, state):
    '''
    Dimension reduction using PCA.
    
    :param str X_std_train: Training data 
    :param str X_std_test: Validation data
    :param int n_features: Number of components
    :return: The training data and validation data with only the new components
    '''
    mod = PCA(n_components = n_features, random_state = state)
    X_std_train = mod.fit_transform(X_std_train)
    X_std_test = mod.transform(X_std_test)
    return(X_std_train, X_std_test) 
    
def mutual(X_std_train, X_std_test, y_train, n_features, colNames):
    '''
    Feature selection using Mutual Information.
    
    :param str X_std_train: Training data 
    :param str X_std_test: Validation data
    :param str y_train: Response to the training data
    :param int n_features: Number of features to be selected
    :return: The training data and validation data with only the selected features
    '''
    leng = X_std_train.shape[1]
    mod = SelectKBest(score_func = mutual_info_classif, k = n_features)
    X_std_train = mod.fit_transform(X_std_train, y_train)
    X_std_test = mod.transform(X_std_test)
    for i in range(leng):
        if mod.get_support()[i] == True:
            print (colNames[i])
    return(X_std_train, X_std_test) 
    
def ICA(X_std_train, X_std_test, n_features, state):
    '''
    Dimension reduction using ICA.
    
    :param str X_std_train: Training data 
    :param str X_std_test: Validation data
    :param int n_features: Number of components
    :param state: Random state
    :return: The training data and validation data with only the new components
    '''
    ica = FastICA(n_components = n_features, random_state = state, fun = 'exp', tol = 0.001)
    X_std_train = ica.fit_transform(X_std_train)
    X_std_test = ica.transform(X_std_test)
    return(X_std_train, X_std_test)