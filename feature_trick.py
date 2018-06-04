# -*- coding: utf-8 -*-
"""
@author: Infaraway
@time: 2018/5/20 21:01
@Function:
"""
import gc
import pandas as pd


class Feature_Trick:

    def __init__(self, data):
        self.data = data

    def set_flag(self, features, new_feat_name):
        """
        标记重复出现位置， 首次出现，最后出现
        :param features: 标记的子集， 选择重复出现的特征
        :param flag_feat_name: 加入data中的特征名称：flag 的名称
        :return:
        """
        self.data.sort_values(['user_id', 'context_timestamp'], inplace=True)
        self.data[new_feat_name] = 0
        pos = self.data.duplicated(subset=features, keep=False)
        self.data.loc[pos, 'click_user_lab'] = 1
        pos = (~self.data.duplicated(subset=features, keep='first')) & self.data.duplicated(subset=features, keep=False)
        self.data.loc[pos, 'click_user_lab'] = 2
        pos = (~self.data.duplicated(subset=features, keep='last')) & self.data.duplicated(subset=features, keep=False)
        self.data.loc[pos, 'click_user_lab'] = 3
        del pos
        gc.collect()

    def time_difference_inDay(self, features, frist_time_diff, last_time_diff):
        """
        同一天点击的时间差
        :param features: 需要考虑的特征子集  subset
        :param frist_time_diff:  第一次点击的时间
        :param last_time_diff:   最后一次点击的时间
        :return:
        """
        self.data.sort_values(['user_id', 'context_timestamp'], inplace=True)

        temp_features = features[:]
        temp_features.append('context_timestamp')
        # 首次出现的时间
        temp = self.data.loc[:, temp_features].drop_duplicates(subset=features, keep='first')
        temp.rename(columns={'context_timestamp': frist_time_diff}, inplace=True)
        self.data = pd.merge(self.data, temp, how='left', on=features)
        self.data[frist_time_diff] = self.data['context_timestamp'] - self.data[frist_time_diff]
        del temp
        gc.collect()
        # 最后出现的时间
        temp = self.data.loc[:, temp_features].drop_duplicates(subset=features, keep='last')
        temp.rename(columns={'context_timestamp': last_time_diff}, inplace=True)
        self.data = pd.merge(self.data, temp, how='left', on=features)
        self.data[last_time_diff] = self.data[last_time_diff] - self.data['context_timestamp']
        del temp
        gc.collect()
        self.data.loc[~self.data.duplicated(subset=features, keep=False), [frist_time_diff, last_time_diff]] = -1

    def time_difference_inNext(self, features):
        """
        获取距离下一次点击的时间差
        :param features: 需要考虑的features
        :return:
        """
        for feat in features:
            self.data[feat + '_next_time_diff'] = 0
            temp = self.data[['context_timestamp', feat, feat + '_next_time_diff']]
            next_time_diff = {}
            for df_line in temp:
                if df_line[1] not in next_time_diff:
                    df_line[2] = -1
                    next_time_diff[df_line[1]] = df_line[0]
                else:
                    df_line[2] = next_time_diff[df_line[1]] - df_line[0]
                    next_time_diff[df_line[1]] = df_line[0]

            self.data[['context_timestamp', feat, feat + '_next_time_diff']] = temp

    def time_difference_inLase(self, features):
        """
        获取距离上一次点击的时间差
        :param features:
        :return:
        """
        for feat in features:
            self.data[feat + '_last_time_diff'] = 0
            temp = self.data[['context_timestamp', feat, feat + '_last_time_diff']]
            last_time_diff = {}
            for df_line in temp:
                if df_line[1] not in last_time_diff:
                    df_line[2] = -1
                    last_time_diff[df_line[1]] = df_line[0]
                else:
                    df_line[2] = df_line[0] - last_time_diff[df_line[1]]
                    last_time_diff[df_line[1]] = df_line[0]

            self.data[['context_timestamp', feat, feat + '_last_time_diff']] = temp


if __name__ == '__main__':
    path = './data/'

    train = pd.read_csv(path + 'train_all.csv')
    test = pd.read_csv(path + 'test_all.csv')

    data = pd.concat([train, test])
    feature_trick = Feature_Trick(data)
    feature_trick.set_flag(['user_id', 'day'], new_feat_name='user_click_time_diff')
    feature_trick.set_flag(['item_id', 'user_id', 'day'], new_feat_name='user_item_click_time_diff')
    feature_trick.set_flag(['item_brand_id', 'user_id', 'day'], new_feat_name='user_brand_click_time_diff')
    feature_trick.set_flag(['shop_id', 'user_id', 'day'], new_feat_name='user_shop_click_time_diff')
    feature_trick.set_flag(['item_city_id', 'user_id', 'day'], new_feat_name='user_city_click_time_diff')

    feature_trick.time_difference_inDay(['item_id', 'day'],
                                        frist_time_diff='item_day_time_diff_first',
                                        last_time_diff='item_day_time_diff_last')
    feature_trick.time_difference_inDay(['user_id', 'day'],
                                        frist_time_diff='user_day_time_diff_first',
                                        last_time_diff='user_day_time_diff_last')
    feature_trick.time_difference_inDay(['item_brand_id', 'user_id', 'day'],
                                        frist_time_diff='user_brand_time_diff_first',
                                        last_time_diff='user_brand_time_diff_last')
    feature_trick.time_difference_inDay(['shop_id', 'user_id', 'day'],
                                        frist_time_diff='user_shop_time_diff_first',
                                        last_time_diff='user_shop_time_diff_last')

    feature_trick.time_difference_inLase(['user_id', 'item_id'])
    feature_trick.time_difference_inNext(['user_id', 'item_id'])

    print(feature_trick.data.shape)
    feature_trick.data.to_csv(path + 'feature_trick.csv', index=False)


