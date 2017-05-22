#!/usr/local/bin/python

import os
import time
import pandas as pd

class split():
    def __init__ (self, file_train, file_test, file_user, file_ad,
                 file_app, file_position, file_user_app, file_user_installed):
        self._train_index = pd.RangeIndex(start = 0, stop = 3200000, step = 1)
        self._test_index = pd.RangeIndex(start = 3200001, stop = 3749528, step = 1)

        self._all_df = pd.read_csv(file_train)
        self._user_df = pd.read_csv(file_user)
        self._position_df = pd.read_csv(file_position)
        self._ad_df = pd.read_csv(file_ad)
        self._app_df = pd.read_csv(file_app)
        self._position_df = pd.read_csv(file_position)
        self._user_app_df = pd.read_csv(file_user_app)
        self._user_installed_df = pd.read_csv(file_user_installed)
        # del self._user_app_df['appID']
        print (time.asctime((time.localtime(time.time()))))

        self._train_df = self._all_df.loc[self._train_index]
        self._train_df = pd.merge(self._train_df, self._user_df, on = 'userID')
        self._train_df = pd.merge(self._train_df, self._ad_df, on = 'creativeID')
        self._train_df = pd.merge(self._train_df, self._position_df, on = 'positionID')
        self._train_df = pd.merge(self._train_df, self._app_df, on = 'appID')
        # self._train_df = pd.merge(self._train_df, self._user_app_df, on = 'appID')
        self._train_df['age'].replace([0], [1])
        self._train_df['appPlatform'].replace([0], [1])
        self._train_df['education'].replace([0], [1])
        self._train_df['gender'].replace([0], [1])

        print (time.asctime((time.localtime(time.time()))))
        self._test_df = self._all_df.loc[self._test_index]
        self._test_df = pd.merge(self._test_df, self._user_df, on = 'userID')
        self._test_df = pd.merge(self._test_df, self._ad_df, on = 'creativeID')
        self._test_df = pd.merge(self._test_df, self._position_df, on = 'positionID')
        self._test_df = pd.merge(self._test_df, self._app_df, on = 'appID')
        # self._test_df = pd.merge(self._test_df, self._user_app_df, on = 'appID')
        
        self._test_df['age'].replace([0], [1])
        self._test_df['appPlatform'].replace([0], [1])
        self._test_df['education'].replace([0], [1])
        self._test_df['gender'].replace([0], [1])

        print (time.asctime((time.localtime(time.time()))))
        self._train_df.to_csv('../data/split_train.csv', index = False)
        self._test_df.to_csv('../data/split_test.csv', index = False)

        print (self._all_df.columns)
        print (self._train_df.columns)
        print (self._test_df.columns)



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
    app = split(file_train, file_test, file_user, file_ad,
                file_app, file_position, file_user_app, file_user_installed)
    print (time.asctime((time.localtime(time.time()))))