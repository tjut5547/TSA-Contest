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

class App():
    def __init__(self, file_train, file_test, file_user, file_ad,
                 file_app, file_position, file_user_app, file_user_installed):
        self._train_df = pd.read_csv(file_train)
        self._test_df = pd.read_csv(file_test)
        self._user_df = pd.read_csv(file_user)
        self._ad_df = pd.read_csv(file_ad)
        self._app_df = pd.read_csv(file_app)
        self._position_df = pd.read_csv(file_position)
        print (time.asctime((time.localtime(time.time()))))

        '''# 获取用户转化率
        self._user = self._train_df['userID'].value_counts()
        self._user_total = self._train_df[self._train_df['label'] == 1]['userID'].value_counts()
        self._user_conversion_rate = pd.DataFrame(data = self._user_total / self._user, columns = ['userID', 'conversionRate'])

        print (self._user[0:5])
        print (self._user_total[0:5])
        print (self._user_conversion_rate)
        print (type(self._user_conversion_rate))
        print (self._user_conversion_rate.columns)
        '''

        # merge train
        self._train_df = pd.merge(self._train_df, self._user_df, on = 'userID')
        self._train_df = pd.merge(self._train_df, self._ad_df, on = 'creativeID')
        self._train_df = pd.merge(self._train_df, self._position_df, on = 'positionID')
        self._train_df = pd.merge(self._train_df, self._app_df, on = 'appID')
        # self._train_df = pd.merge(self._train_df, self._user_conversion_rate, on = 'userID')
        self._train_df['age'].replace([0], [1])
        self._train_df['appPlatform'].replace([0], [1])
        self._train_df['education'].replace([0], [1])
        self._train_df['gender'].replace([0], [1])

        #  merge test
        self._test_df = pd.merge(self._test_df, self._user_df, on = 'userID')
        self._test_df = pd.merge(self._test_df, self._ad_df, on = 'creativeID')
        self._test_df = pd.merge(self._test_df, self._position_df, on = 'positionID')
        self._test_df = pd.merge(self._test_df, self._app_df, on = 'appID')
        # self._test_df = pd.merge(self._test_df, self._user_conversion_rate, on = 'userID')
        self._test_df['age'].replace([0], [1])
        self._test_df['appPlatform'].replace([0], [1])
        self._test_df['education'].replace([0], [1])
        self._test_df['gender'].replace([0], [1])

    def _removeSkewness(self):
        # Store target variable and remove skewness
        target = self._train_df['label']
        instanceID = self._test_df['instanceID']
        self._Result_Train = target
        self._test_instanceID = instanceID
        
        del self._train_df['label']
        del self._train_df['conversionTime']
        del self._test_df['instanceID']
        del self._test_df['label']

    def _NormalizerData(self):
        print ("columns of train : ", self._train_df.columns)
        print ("columns of test :  ", self._test_df.columns)

        index_train = self._train_df.loc[self._train_df.index]
        index_test = self._test_df.loc[self._test_df.index]
        self._xTrain = index_train.values # 获得训练集数据, array
        self._xTest = index_test.values # 获得测试集数据, array

    def _xboost(self):
        # Fitting the model and predicting using xgboost
        regr = xgb.XGBRegressor(colsample_bytree=0.4,
                gamma=0.05,
                learning_rate=0.16,
                max_depth=6,
                # min_child_weight=1.5,
                n_estimators=85,
                reg_alpha=1.2,
                # reg_lambda=1.5,
                subsample=1,
                objective='binary:logistic',
                silent=True)

        regr.fit(self._xTrain,
                 self._Result_Train,
                 eval_metric = 'logloss')
        y_pred_xgb = regr.predict(self._xTest)
        return  y_pred_xgb

    def _submission(self, y_final):
        # Preparing for submissions
        submission_df = pd.DataFrame(data= {'instanceID' : self._test_instanceID, 'prob': y_final})
        submission_df.to_csv('submission.csv', index=False)

    def main(self):
        self._removeSkewness()
        self._NormalizerData()
        y_final = self._xboost()
        self._submission(y_final)

class App_Train():
    def __init__(self, file_train, file_test):
        self._train_df = pd.read_csv(file_train)
        self._test_df = pd.read_csv(file_test)
        self._yTrain = self._train_df['label']
        self._yTest = self._test_df['label']

        del self._train_df['label']
        del self._train_df['conversionTime']
        del self._test_df['label']
        del self._test_df['conversionTime']

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
        self._xTrain = index_train.values
        self._xTest = index_test.values

    def _xboost(self):
        num_tree = [100]#, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270]
        for num in num_tree:
            print ('********************************************************')
            print (time.asctime((time.localtime(time.time()))))
            regr = xgb.XGBRegressor(colsample_bytree=0.4,
                    # gamma=gam,
                    learning_rate=0.16,
                    max_depth=6,
                    # min_child_weight=1.5,
                    n_estimators=115,
                    reg_alpha=1.2,
                    # reg_lambda=1.5,
                    subsample=1,
                    objective='binary:logistic',
                    silent=True)
            regr.fit(self._xTrain,
                     self._yTrain,
                     eval_metric = 'logloss')
            result = regr.predict(self._xTest)
            print ("LOG LOSS OF THIS ALGO : ", num, self._log_loss(self._yTest, result))

    def _log_loss(self, act, pred):
        epsilon = 1e-15
        pred = sp.maximum(epsilon, pred)
        pred = sp.minimum(1-epsilon, pred)
        ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
        ll = ll * -1.0/len(act)
        return ll

    def main(self):
        self._xboost()
        # print ("LOG LOSS OF THIS ALGO : ", self._log_loss(self._yTest, result))

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
    file_Train = path + 'split_train.csv'
    file_Test = path + 'split_test.csv'
    
    app = App(file_train, file_test, file_user, file_ad, file_app, file_position, file_user_app, file_user_installed)
    # app = App_Train(file_Train, file_Test)
    app.main()
    print (time.asctime((time.localtime(time.time()))))
