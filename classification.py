# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 2018

@author: Alise Midtfjord
"""


# Classification using the thirteen classifiers givens in the script "functions".
"------------------Classification------------------"

def classify(input_excel, ark, y_navn, n_features):
    '''
    Classification using thirteen classifiers defined in the script "functions". 
    Uses 4-folds-CV ten times with different splits each times. Uses given number of
    features and chosen feature selector.
    
    :param str input_excel: The name of the excel-file with the dataset
    :param str ark: The name of the sheet with the dataset
    :param str y_navn: The name of the column with the response
    :param int n_features: Number of features to use in the models
    :return: Matrix with the AUC of all classificatons and matrix with the selected 
             features
    '''
    # Reads the excel-file
    xls = pd.ExcelFile(input_excel) 
    data_raw_df = pd.read_excel(xls, sheetname=ark, index_col = 0)
    
    # Creates the result-matrix
    results = [[],[],[],[],[],[],[],[],[],[]] 
    for i in range(0,10):
        results[i] = np.zeros((13,4))

    # Splits the respons y and the variables X and sets the random states
    y_name = y_navn
    y = data_raw_df[y_name].values 
    X= data_raw_df.drop(y_name,1) 
    colNames = list(X.columns) 
    states =  [108, 355, 44, 129, 111, 362, 988, 266, 82,581] # Change to wanted seeds 
    features = []
    stdsc = StandardScaler()

    # Splits the dataset into the 10*4 folders and selects features and uses the classifers
    for k in range(0,10): 
        i = 0
        state = states[k]
        cv = StratifiedKFold(n_splits=4, random_state = state, shuffle = True) 
        for train, test in cv.split(X, y):
            print(k,i)
            X_train = X.iloc[train]
            X_test = X.iloc[test]
            y_train = y[train]
            y_test = y[test]
            X_std_train = stdsc.fit_transform(X_train) 
            X_std_test = stdsc.transform(X_test) 
            X_std_train, X_std_test, features = relieff(X_std_train, X_std_test, y_train,n_features, colNames, features)
            
            model = logrel1(X_std_train, X_std_test, state)
            print('Test score L1-Logistic regression:', model.score(X_std_test,y_test))
            results[k][0,i] = model.score(X_std_test,y_test)
            model = logrel2(X_std_train, X_std_test, state) 
            print('Test score L2-Logistic regression:', model.score(X_std_test,y_test))
            results[k][1,i] = model.score(X_std_test,y_test) 
            model = rf(X_train,y_train, state)
            print('Test score Random forest:', model.score(X_test,y_test))
            results[k][2,i] = model.score(X_test,y_test)  
            model = knn(X_std_train, y_train)
            print('Test score KNN:', model.score(X_std_test,y_test))
            results[k][3,i] = model.score(X_std_test,y_test)
            model = adaboostlog(X_std_train,  y_train,  state = state)
            print('Test score AdaBoost:', model.score(X_std_test,y_test))
            results[k][4,i] = model.score(X_std_test,y_test)
            model = decisiontree(X_std_train,y_train, state)
            print('Test score Decision Tree:', model.score(X_test,y_test))
            results[k][5,i] = model.score(X_std_test,y_test)
            model = gnb(X_std_train, y_train,  state = state)
            print('Test score GNB:', model.score(X_std_test,y_test))
            results[k][6,i] = model.score(X_std_test,y_test)
            model = lda(X_std_train, y_train)
            print('Test score Linear LDA:', model.score(X_std_test,y_test))
            results[k][7,i] = model.score(X_std_test,y_test)
            model = qda(X_std_train, y_train)
            print('Test score QDA:', model.score(X_std_test,y_test))
            results[k][8,i] = model.score(X_std_test,y_test)
            model = nnet(X_std_train, y_train,  state = state)
            print('Test score Neural network:', model.score(X_std_test,y_test))
            results[k][9,i] = model.score(X_std_test,y_test)
            model =  mars(X_std_train, y_train)
            print('Test score MARS:', model.score(X_std_test,y_test))
            results[k][10,i] = model.score(X_std_test,y_test)
            model =  plsr(X_std_train, y_train)
            print('Test score PLSR:', model.score(X_std_test,y_test))
            results[k][11,i] = model.score(X_std_test,y_test)
            model = svc(X_std_train, X_std_test, y_train, y_test, state)
            print('Test score SVC:', model.score(X_std_test,y_test))
            results[k][12,i] = model.score(X_std_test,y_test)
            model = linearsvc(X_std_train, y_train, state)
            print('Test score Linear SVC:', model.score(X_std_test,y_test))
            results[k][13,i] = model.score(X_std_test,y_test)
            i += 1
            
    return (results, features)


"------------------Number of features------------------"


def n_features(input_excel, ark, y_navn):
    '''
    Classification using thirteen classifiers defined in the script "functions".  
    Uses 4-folds-CV ten times with different splits each times. Tries different number of features from 1-20.
    
    :param str input_excel: The name of the excel-file with the dataset
    :param str ark: The name of the sheet with the dataset
    :param str y_navn: The name of the column with the response
    :return: Matrix with the AUC of all classificatons for different number of features and matrix with the selected features
    '''
    
    # Reads the excel-file
    xls = pd.ExcelFile(input_excel)
    data_raw_df = pd.read_excel(xls, sheetname=ark, index_col = 0)

    # Creates the result-matrix
    results = [[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]],
               [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]]
    for j in range(0,2):
        for i in range(0,20):
            results[j][i] = np.zeros((13,4))

    # Splits the respons y and the variables X and sets the random states
    y_name = y_navn
    y = data_raw_df[y_name].values 
    X= data_raw_df.drop(y_name,1) 
    stdsc = StandardScaler()
    colNames = list(X.columns)  
    states = [209, 979]  # Change to wanted seeds 
    features = []
    stdsc = StandardScaler()

    # Splits the dataset into the 2*4 folders and selects features and uses the classifers
    # for 1-20 number of features-
    for l in range(0,2):
        state = states[l]
        cv = StratifiedKFold(n_splits=4, random_state = state, shuffle = True)
        for k in range(0,20): 
            i = 0
            n_features = k + 1
            for train, test in cv.split(X, y):
                print(k,i)
                X_train = X.iloc[train]
                X_test = X.iloc[test]
                y_train = y[train]
                y_test = y[test]
                X_std_train = stdsc.fit_transform(X_train) 
                X_std_test = stdsc.transform(X_test) 
                X_std_train, X_std_test = relieff(X_std_train, X_std_test, y_train,n_features, colNames, features)
                
                model = logrel1(X_std_train, X_std_test, y_train, y_test, state)
                results[l][k][0,i] = model.score(X_std_test,y_test)
                model = logrel2(X_std_train, X_std_test, y_train, y_test, state) 
                results[l][k][1,i] = model.score(X_std_test,y_test) 
                model = rf(X_train,y_train, X_test, y_test, state)
                results[l][k][2,i] = model.score(X_test,y_test)  
                model = knn(X_std_train, X_std_test, y_train, y_test)
                results[l][k][3,i] = model.score(X_std_test,y_test)
                model = adaboostlog(X_std_train, X_std_test, y_train, y_test, state = state)
                results[l][k][4,i] = model.score(X_std_test,y_test)
                model = decisiontree(X_std_train,y_train, X_std_test, y_test, state)
                results[l][k][5,i] = model.score(X_std_test,y_test)
                model = gnb(X_std_train, X_std_test, y_train, y_test, state = state)
                results[l][k][6,i] = model.score(X_std_test,y_test)
                model = lda(X_std_train, X_std_test, y_train, y_test)
                results[l][k][7,i] = model.score(X_std_test,y_test)
                model = qda(X_std_train, X_std_test, y_train, y_test)
                results[l][k][8,i] = model.score(X_std_test,y_test)
                model = nnet(X_std_train, X_std_test, y_train, y_test, state = state)
                results[l][k][9,i] = model.score(X_std_test,y_test)
                model =  mars(X_std_train, X_std_test, y_train, y_test)
                results[l][k][10,i] = model.score(X_std_test,y_test)
                model =  plsr(X_std_train, X_std_test, y_train, y_test)
                results[l][k][11,i] = model.score(X_std_test,y_test)
                model = svc(X_std_train, X_std_test, y_train, y_test, state)
                results[l][k][12,i] = model.score(X_std_test,y_test)
                model = linearsvc(X_std_train, X_std_test, y_train, y_test, state)
                results[l][k][13,i] = model.score(X_std_test,y_test)
                i += 1
                
    return (results)

"------------------Main------------------"


if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    
    # import modules and functions from the file functions.py
    from sklearn.model_selection import (StratifiedKFold, GridSearchCV)
    from sklearn.preprocessing import StandardScaler
    from functions import (logrel1, logrel2, rf, knn, adaboostlog, decisiontree, gnb, lda,
                             qda, nnet, mars, plsr, svc, linearsvc, relieff, rf_selection, 
                             logrel1_selection, lda_selection, pca, mutual, ICA)
    

    # Insert the right names
    input_excel = "dataset_name.xlsx"
    ark = 'sheet_name'
    y_navn = 'response_name'
    
    # Choose method. Method = 0 for classification with a spesific number of features, or method = 1 for
    # trying 1-20 different number of features.
    method = 1 
    if method == 0:
        n_features = 2 # Change to wanted number of features
        results, features = classify(input_excel, ark, y_navn, n_features)
    elif method == 1:
        results = n_features(input_excel, ark, y_navn)
        
    