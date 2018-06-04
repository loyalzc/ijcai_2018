# -*- coding: utf-8 -*-
"""
@author: Infaraway
@time: 2018/5/20 21:56
@Function:
"""
import datetime
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class Feature_Base:

    def __init__(self, data):
        self.data = data

    def base_process(self):
        self.data['time'] = pd.to_datetime(self.data.context_timestamp, unit='s')
        self.data['time'] = self.data['time'].apply(lambda x: x + datetime.timedelta(hours=8))
        self.data['day'] = self.data['time'].apply(lambda x: int(str(x)[8:10]))
        self.data['hour'] = self.data['time'].apply(lambda x: int(str(x)[11:13]))
        self.data['minute'] = self.data['time'].apply(lambda x: int(str(x)[14:16]))

        self.data['maphour'] = self.data['hour'].map(self._map_hour)
        self.data['mapmin'] = self.data['minute'] % 15 + 1

        data_item_category = self.data.item_category_list.str.split(';', expand=True).add_prefix('item_category_')

        for i in range(3):
            self.data['item_category_' + str(i)] = data_item_category['item_category_' + str(i)]
        del self.data['item_category_0']
        self.data['item_category_1'] = self.data['item_category_1'].apply(int)
        self.data['item_category_2'].fillna(value=0, inplace=True)
        self.data['item_category_2'] = self.data['item_category_2'].apply(int)

        features_label = ['item_category_1', 'item_category_2', 'context_id', 'item_brand_id', 'item_city_id', 'item_id',
                 'user_id', 'shop_id','context_page_id', 'shop_star_level', 'user_age_level', 'user_occupation_id',
                 'user_star_level']

        features_score = ['shop_score_service', 'shop_review_positive_rate', 'shop_score_delivery',
                       'shop_score_description']

        for col in features_label:
            col_encoder = LabelEncoder()
            col_encoder.fit_transform(self.data[col])

        for col in features_score:
            self.data[col] = round(self.data[col], 3)

        del self.data['predict_category_property']
        del self.data['item_property_list']

    def _map_hour(self, s):
        if s < 6:
            return 1
        elif s < 12:
            return 2
        elif s < 18:
            return 3
        else:
            return 4


if __name__ == '__main__':
    path = './data/'

    # 读取全部数据
    train = pd.read_table(path + 'round2_train.txt', sep=" ")
    test_a = pd.read_table(path + 'round2_ijcai_18_test_a_20180425.txt', sep=" ")
    test_b = pd.read_table(path + 'round2_ijcai_18_test_b_20180510.txt', sep=" ")
    test = test_a.append(test_b)
    data = pd.concat([train, test])

    feature_base = Feature_Base(data)

    train_all = data[:len]
    test_all = data[len:]
    train_all.to_csv(path + 'train_all.csv', index=False)
    test_all.to_csv(path + 'test_all.csv', index=False)