# -*- coding: utf-8 -*-
"""
@author: Infaraway
@time: 2018/5/20 22:14
@Function:
"""
import numpy as np
import pandas as pd


class Feature_Statistic:

    def __init__(self, data):
        self.data = data


    def expose_feature(self, group_features, count_feaure):
        """
        考虑 userid itmeid 的曝光度
        :param group_features:  需要联合考虑曝光度的特征组
        :param count_feaure:  需要count feature的特征
        :return:
        """
        feat_string = '_'.join(group_features)

        expose = self.data.groupby(group_features)['instance_id'].nunique().to_frame()
        expose.columns = ['count_feature']
        expose = expose.reset_index()
        expose[feat_string + '_expose'] = round(
            expose.count_feature / expose.groupby(count_feaure).count_feature.transform(np.sum), 5)

        new_features = group_features[:]
        new_features.append(feat_string + '_expose')
        expose = expose[new_features]

        self.data = self.data.merge(expose, how='left', on=group_features)

    def average_feature(self, group_features, avg_feature):
        """
        平均值特征
        :param group_feature: 用于分组的特征
        :param avg_feature: 求均值的特征
        :return:
        """
        feat_string = '_'.join(group_features)
        grouped = pd.DataFrame(self.data.groupby(group_features)[avg_feature].nunique()).reset_index()

        new_features = group_features[:]
        new_features.append(feat_string + '_avg_' + avg_feature)
        grouped.columns = new_features
        self.data = self.data.merge(grouped, how='left', on=group_features)

    def active_feature(self, group_features, act_feature):
        """
        活跃值特征
        :param group_feature: 用于分组的特征
        :param act_feature: 求活跃的特征
        :return:
        """
        feat_string = '_'.join(group_features)
        grouped = pd.DataFrame(self.data.groupby(group_features)[act_feature].nunique()).reset_index()

        new_features = group_features[:]
        new_features.append(feat_string + '_active_' + act_feature)
        grouped.columns = new_features
        self.data = self.data.merge(grouped, how='left', on=group_features)


if __name__ == '__main__':
    path = './data/'

    train = pd.read_csv(path + 'ntrain_all.csv')
    test = pd.read_csv(path + 'ntest_all.csv')
    print(train['day'].unique())
    data = pd.concat([train, test])

    feature_statistic = Feature_Statistic(data)
    feature_statistic.expose_feature(group_features=['item_id', 'hour'], count_feaure='item_id')
    feature_statistic.expose_feature(group_features=['item_id', 'maphour'], count_feaure='item_id')
    feature_statistic.expose_feature(group_features=['user_id', 'hour'], count_feaure='user_id')
    feature_statistic.expose_feature(group_features=['user_id', 'maphour'], count_feaure='user_id')
    feature_statistic.expose_feature(group_features=["user_id", "context_timestamp"], count_feaure='user_id')

    feature_statistic.average_feature(group_features=['user_id'], avg_feature='hour')
    feature_statistic.average_feature(group_features=['item_id'], avg_feature='hour')
    feature_statistic.average_feature(group_features=['item_brand_id'], avg_feature='hour')
    feature_statistic.average_feature(group_features=['shop_id'], avg_feature='hour')

    feature_statistic.average_feature(group_features=['user_id'], avg_feature='user_age_level')
    feature_statistic.average_feature(group_features=['item_id'], avg_feature='user_age_level')
    feature_statistic.average_feature(group_features=['item_brand_id'], avg_feature='user_age_level')
    feature_statistic.average_feature(group_features=['shop_id'], avg_feature='user_age_level')

    feature_statistic.active_feature(group_features=['user_id'], act_feature='hour')
    feature_statistic.active_feature(group_features=["item_category_list", "day"], act_feature='item_id')
    feature_statistic.active_feature(group_features=["user_id", "day"], act_feature='item_city_id')
    feature_statistic.active_feature(group_features=["user_id", "day", "hour"], act_feature='item_city_id')

    feature_statistic.active_feature(group_features=["item_id", "day"], act_feature='user_id')
    feature_statistic.active_feature(group_features=["shop_id", "day"], act_feature='user_id')
    feature_statistic.active_feature(group_features=["item_brand_id", "day"], act_feature='user_id')
    feature_statistic.active_feature(group_features=["item_category_list", "day"], act_feature='user_id')
    feature_statistic.active_feature(group_features=["item_id", "day", "hour"], act_feature='user_id')
    feature_statistic.active_feature(group_features=["shop_id", "day", "hour"], act_feature='user_id')

    feature_statistic.active_feature(group_features=["user_id", "day"], act_feature='shop_id')
    feature_statistic.active_feature(group_features=["user_id", "day"], act_feature='item_brand_id')
    feature_statistic.active_feature(group_features=["user_id", "day", "hour"], act_feature='item_brand_id')

    feature_statistic.data.to_csv('feature_statistic.csv', index=False)

