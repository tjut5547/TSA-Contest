#!/usr/local/bin/python

import time
import sys, pickle
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from scipy.stats import skew
from sklearn import decomposition
from sklearn.linear_model import LogisticRegression

Columns = ['creativeID', 'positionID', 'connectionType', 'telecomsOperator',
           'age', 'gender', 'education', 'marriageStatus', 'haveBaby', 'hometown',
           'residence', 'adID', 'camgaignID', 'advertiserID', 'appID', 'appPlatform',
           'sitesetID', 'positionType', 'appCategory']

class LR_TEST():
    def __init__(self, file_train, file_test):
        self._train_df = pd.read_csv(file_train)
        self._test_df = pd.read_csv(file_test)

        self._train_df['age'].replace([0], [1])
        self._train_df['appPlatform'].replace([0], [1])
        self._train_df['education'].replace([0], [1])
        self._train_df['gender'].replace([0], [1])
        self._test_df['age'].replace([0], [1])
        self._test_df['appPlatform'].replace([0], [1])
        self._test_df['education'].replace([0], [1])
        self._test_df['gender'].replace([0], [1])

        index_train = self._train_df.loc[self._train_df.index]
        index_test = self._test_df.loc[self._test_df.index]

        self._yTrain = self._train_df['label']
        self._yTest = self._test_df['label']
        # self._xTrain = index_train.values # 获得训练集数据, array
        # self._xTest = index_test.values # 获得测试集数据, array

    def _dummyCreate(self):
        self._train_df = pd.get_dummies(self._train_df, columns = Columns, sparse = True)
        self._test_df = pd.get_dummies(self._test_df, columns = Columns, sparse = True)

        pickle.dump((self._train_df), open('../pickle/train.pickle', 'wb'))
        pickle.dump((self._test_df), open('../pickle/test.pickle', 'wb'))
        print ("number of train features : ", len(self._train_df.columns))
        print ("number of test features : ", len(self._test_df.columns))

        # self._train_df = self._train_df.fillna(self._train_df.mean())
        # self._test_df = self._test_df.fillna(self._test_df.mean())

    def main(self):
        self._dummyCreate()

class LR():
    def __init__(self, file_train, file_test, file_user, file_ad,
                 file_app, file_position, file_user_app, file_user_installed):
        self._train_df = pd.read_csv(file_train)
        self._test_df = pd.read_csv(file_test)
        self._user_df = pd.read_csv(file_user)
        self._ad_df = pd.read_csv(file_ad)
        self._app_df = pd.read_csv(file_app)
        self._position_df = pd.read_csv(file_position)
        # self._user_app_df = pd.read_csv(file_user_app)
        # self._user_installed_df = pd.read_csv(file_user_installed)
        print (time.asctime((time.localtime(time.time()))))
        
        # merge train
        self._train_df = pd.merge(self._train_df, self._user_df, on = 'userID')
        self._train_df = pd.merge(self._train_df, self._ad_df, on = 'creativeID')
        self._train_df = pd.merge(self._train_df, self._position_df, on = 'positionID')
        self._train_df = pd.merge(self._train_df, self._app_df, on = 'appID')
        self._train_df['age'].replace([0], [1])
        self._train_df['appPlatform'].replace([0], [1])
        self._train_df['education'].replace([0], [1])
        self._train_df['gender'].replace([0], [1])

        # merge test
        self._test_df = pd.merge(self._test_df, self._user_df, on = 'userID')
        self._test_df = pd.merge(self._test_df, self._ad_df, on = 'creativeID')
        self._test_df = pd.merge(self._test_df, self._position_df, on = 'positionID')
        self._test_df = pd.merge(self._test_df, self._app_df, on = 'appID')
        self._test_df['age'].replace([0], [1])
        self._test_df['appPlatform'].replace([0], [1])
        self._test_df['education'].replace([0], [1])
        self._test_df['gender'].replace([0], [1])
    
    def _removeSkewness(self):
        # Store target variable and remove skewness
        self._yTrain = self._train_df['label']
        self._instanceID = self._test_df['instanceID']
        
        del self._train_df['userID']
        del self._train_df['label']
        del self._train_df['conversionTime']
        del self._train_df['clickTime']
        del self._test_df['clickTime']
        del self._test_df['label']
        del self._test_df['userID']
        del self._test_df['instanceID']

    def _dummyCreate(self):
        self._train_df = pd.get_dummies(self._train_df, columns = Columns, sparse = True)
        print (time.asctime((time.localtime(time.time()))))
        self._test_df = pd.get_dummies(self._test_df, columns = Columns, sparse = True)
        pickle.dump((self._train_df), open('../pickle/train.pickle', 'wb'))
        pickle.dump((self._test_df), open('../pickle/test.pickle', 'wb'))
        print (time.asctime((time.localtime(time.time()))))
        print ("number of train features : ", len(self._train_df.columns))
        print ("number of test features : ", len(self._test_df.columns))

    def _prediction(self):
        index_train = self._train_df.loc[self._train_df.index]
        index_test = self._test_df.loc[self._test_df.index]
        self._xTrain = index_train.values # 获得训练集数据, array
        self._xTest = index_test.values # 获得测试集数据, array

        lr = LogisticRegression().fit(self._xTrain)
        self._yTest = lr.predict_proba(self._yTrain)
    
    def main(self):
        self._removeSkewness()
        self._dummyCreate()
        # self._NormalizerData()
        # y_final = self._xboost()
        # self._submission(y_final)

if __name__ == '__main__':
    print (time.asctime((time.localtime(time.time()))))
    path = '../data/'
    file_train = path + 'train.csv'
    file_test = path + 'test.csv'
    file_user = path + 'user.csv'
    file_ad = path + 'ad.csv'
    file_app = path + 'app_categories.csv'
    file_position = path + 'position.csv'
    file_user_app = path + 'user_app_actions.csv'
    file_user_installed = path + 'user_installedapps.csv'

    app = LR(file_train, file_test, file_user, file_ad,
                 file_app, file_position, file_user_app, file_user_installed)
    file_Train = path + 'split_train.csv'
    file_Test = path + 'split_test.csv'
    # app = LR_TEST(file_Train, file_Test)
    app.main()
    print (time.asctime((time.localtime(time.time()))))