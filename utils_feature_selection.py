# -*- coding: utf-8 -*-
"""
@author: Infaraway
@time: 2018/5/15 15:06
@Function:
"""
import random

import lightgbm as lgb
import time
from sklearn.metrics import auc, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import numpy as np

class Feature_selection:

    def __init__(self, data, base_feature, test_feature):
        """
        :param data:
        :param base_feature: 基本特征，不需要进行选择
        :param test_feature:  需要进行特征选择的特征
        """
        self.data = data
        self.test_features = test_feature
        self.base_feature = base_feature
        self.best_features = []
        print('len base_feature: ', len(self.base_feature), '  len test feature: ', len(self.test_features))

    def _base_classifier(self):
        clf = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
                                 max_depth=-1, n_estimators=100, objective='binary',
                                 subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
                                 learning_rate=0.05, min_child_weight=50, random_state=2018, n_jobs=20)
        return clf

    def _get_features_score(self, features):
        # 计算三次结果的最好得分作为最终的结果，尽量消除由随机带来的误差
        best_scores = []
        model = self._base_classifier()
        X_train, X_test, y_train, y_test = train_test_split(self.data[features].values, self.data['label'].values,
                                                            test_size=0.5, random_state=2018)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1024)
        for i, (train_i, test_i) in enumerate(skf.split(X_train, y_train)):
            model.fit(X_train[train_i], y_train[train_i])
            y_pred = model.predict_proba(X_train[test_i])[:, 1]
            score = roc_auc_score(y_train[test_i], y_pred)
            best_scores.append(score)
        return np.mean(best_scores)

    def find_best_features(self):
        feat_num = len(self.test_features)

        try:
            for num_round in range(5):
                test_features = self.test_features
                now_feature = self.base_feature
                now_scores = self._get_features_score(self.base_feature)
                print('-------------now best features len: ', len(self.best_features), 'best scores: ', now_scores, 'this is round 5 \ :', num_round)
                for i in range(feat_num):
                    start = time.time()
                    feat = test_features[random.randint(0, len(test_features) - 1)]
                    # print('    --- this is feature:', feat, '  ---this is: ', i, ' / ', len(self.test_features))
                    now_feature.append(feat)
                    new_scores = self._get_features_score(now_feature)
                    print('        --- now_scores:', now_scores, '----now - best  score : ', new_scores - now_scores)
                    if (new_scores - now_scores) > 0.0001:
                        now_scores = new_scores
                        if feat not in self.best_features:
                            self.best_features.append(feat)
                            print('        ---now best features len: ', len(self.best_features), 'best scores: ', now_scores)
                    else:
                        now_feature.remove(feat)

                    test_features.remove(feat)
                    print('    ----time min:', (time.time() - start) / 60)
        except Exception as e:
            print('************************** Error ************************************')
            print(e)
            print('*********************************************************************')
        finally:
            print('       ---best features len: ', len(self.best_features))
            print(self.best_features)


def get_data():
    print('------------------------read data :')
    data = []
    return data


if __name__ == '__main__':
    get_data()