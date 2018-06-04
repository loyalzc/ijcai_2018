# -*- coding: utf-8 -*-
"""
@author: Infaraway
@time: 2018/4/17 17:39
@Function:
"""
import gc
from sklearn import preprocessing

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder


class Feature_Cross:

    def __init__(self, base_data, cross_feature):
        self.base_data = base_data
        self.cross_feature = cross_feature

    def cross_2feature(self, feat1, feat2, agg_feat):
        """两个的交叉占比"""
        item = self.base_data.groupby(feat1, as_index=False)[agg_feat].agg({feat1 + '_count': 'count'})
        self.base_data = pd.merge(self.base_data, item, on=[feat1], how='left')

        itemcnt = self.base_data.groupby([feat1, feat2], as_index=False)[agg_feat].agg({feat1 + feat2: 'count'})
        self.base_data = pd.merge(self.base_data, itemcnt, on=[feat1, feat2], how='left')
        self.cross_feature[feat1 + '_' + feat2 + '_prob'] = self.base_data[feat1 + feat2] / self.base_data[feat1 + '_count']
        self.cross_feature[feat1 + '_' + feat2 + '_prob'] = pd.qcut(self.cross_feature[feat1 + '_' + feat2 + '_prob'], 10, duplicates='drop')
        # self.cross_feature[feat1 + '_' + feat2 + '_prob'] = self.cross_feature[feat1 + '_' + feat2 + '_prob'].round(7)

        del self.base_data[feat1 + '_count']
        del self.base_data[feat1 + feat2]
        gc.collect()
        print('    ----cross feature: %s  and %s' % (feat1, feat2))

    def cross_3feature(self, feat1, feat2, feat3, agg_feat):
        """三个特征的交叉占比"""
        item = self.base_data.groupby([feat1, feat2], as_index=False)[agg_feat].agg({feat1 + feat2 + '_count': 'count'})
        self.base_data = pd.merge(self.base_data, item, on=[feat1, feat2], how='left')

        itemcnt = self.base_data.groupby([feat1, feat2, feat3], as_index=False)[agg_feat].agg({feat1 + feat2 + feat3: 'count'})
        self.base_data = pd.merge(self.base_data, itemcnt, on=[feat1, feat2, feat3], how='left')
        self.cross_feature[feat1 + '_' + feat2 + '_' + feat3 + '_prob'] = \
            self.base_data[feat1 + feat2 + feat3] / self.base_data[feat1 + feat2 + '_count']
        # self.cross_feature[feat1 + '_' + feat2 + '_' + feat3 + '_prob'] = \
        # pd.cut(self.cross_feature[feat1 + '_' + feat2 + '_' + feat3 + '_prob'], 10, labels=range(10))

        self.cross_feature[feat1 + '_' + feat2 + '_' + feat3 + '_prob'] = \
            self.cross_feature[feat1 + '_' + feat2 + '_' + feat3 + '_prob'].round(7)

        del self.base_data[feat1 + feat2 + '_count']
        del self.base_data[feat1 + feat2 + feat3]
        gc.collect()
        print('    ----cross feature: %s  %s  and %s' % (feat1, feat2, feat3))


    def combine(self, feat1, feat2):
        print('    ----cross feature: %s and %s' % (feat1, feat2))
        self.base_data[feat1 + '_' + feat2] = self.base_data[feat1] + '_' + self.base_data[feat2]
        self.cross_feature[feat1 + '_' + feat2] = LabelEncoder().fit_transform(self.base_data[feat1 + '_' + feat2])
        self.cross_feature[feat1 + '_' + feat2] = self.cross_feature[feat1 + '_' + feat2].apply(int)


if __name__ == '__main__':
    print('--------------------cross feature----------------------')

