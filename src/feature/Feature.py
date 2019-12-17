import sys
import random
import pandas as pd
import os
import numpy as np


class Feature():

    def __init__(self):
        self.columns = ['index', 'field_id', 'type', 'method']
        self.feature_num = 0
        self.cliplist = []
        self.feature_columns = []
        self.category_columns = []
        self.continous_columns = []
        self.feature_table = {}
        self.q_min = {}
        self.q_max = {}
        self.dicts = {}
        self.dists = {}
        self.cdf_bounary = {}
        self.sample = None
        self.tmp = False

    def load(self, feature_dir):
        f_num = 0
        c_num = 0
        q_num = 0
        with open(feature_dir, 'r') as f:
            for line in f:
                tz = line.strip('\n').split('\t')
                index = int(tz[0])
                filed_id = int(tz[1])
                f_name = tz[2]
                f_type = tz[3]
                f_method = tz[4]
                self.feature_table[f_name] = dict(zip(self.columns, [index, filed_id, f_type, f_method]))
                if f_type == 'q':
                    self.continous_columns.append(f_name)
                    self.q_min[f_name] = sys.maxsize
                    self.q_max[f_name] = -sys.maxsize
                else:
                    self.category_columns.append(f_name)
                    self.dicts[f_name] = {}
                self.feature_columns.append(f_name)
                f_num = f_num + 1
            self.feature_num = f_num

    # 对gbdt输出的中间变量做处理，后续作为LR，FM的输入
    def tmp_config(self):
        datafile = "./data/gbdt_tmp/train.txt"
        with open(datafile, 'r') as f:
            for i, line in enumerate(f):
                if i == 0:
                    sample = line
        tz = sample.strip('\n').split('\t')[1:]
        f_name = ["C" + str(i + 1) for i in range(len(tz))]
        for i, item in enumerate(f_name):
            self.feature_table[item] = dict(zip(self.columns, [i, i, 'c', 'onehot']))
            self.dicts[item] = {}
        self.category_columns = f_name
        self.feature_columns = f_name
        self.tmp = True

    def getSample(self, featurePath):
        sample = [[0.0]]
        with open(featurePath, 'r') as f:
            for line in f:
                tz = line.strip().split('\t')
                if str(tz[3]) == 'q':
                    sample.append([0.0])
                if str(tz[3]) == 'c':
                    sample.append(['0'])
        return sample

    def build(self, input_dir, cutoff, percent):
        datafile = input_dir + '/train.txt'
        if self.tmp == False:
            self.cliplist = Feature.continous_clip(self.feature_columns, self.continous_columns, datafile, percent)
            # self.cliplist =  None
        with open(datafile, 'r') as f:
            for indx, line in enumerate(f):
                features = line.rstrip('\n').split('\t')
                # 离散特征
                for item in self.category_columns:
                    indx = self.feature_table[item]['index'] + 1
                    if features[indx] not in ['\\N', '', r'\N']:
                        if features[indx] not in self.dicts[item].keys():
                            self.dicts[item][features[indx]] = 1
                        else:
                            self.dicts[item][features[indx]] += 1
                # 连续特征
                if self.tmp == False:
                    Feature.continous_build(self.q_min,
                                            self.q_max,
                                            self.cliplist,
                                            features,
                                            self.feature_table,
                                            self.continous_columns)
            self.dicts = Feature.category_build(self.dicts, self.category_columns, 20)

    def gen(self, column, val):
        if column in self.category_columns:
            return Feature.category_gen(self.dicts, column, val)
        else:
            return Feature.continous_gen(column, self.q_min, self.q_max, val)

    def get_bounary(self, input_dir):
        input_path = input_dir + '/train.txt'
        cdf_table = {}
        df = pd.read_csv(input_path, sep='\t', header=None)
        df.columns = ['label'] + self.feature_columns
        for item in self.continous_columns:
            feature_column = df[item].tolist()
            self.cdf_bounary[item] = Feature.cdf_onehot(feature_column)
        del df

    @staticmethod
    def category_build(dicts, category_columns, cutoff):
        for item in category_columns:
            if dicts[item] != {}:
                dicts[item] = filter(lambda x: x[1] >= cutoff, dicts[item].items())
                dicts[item] = sorted(dicts[item], key=lambda x: (-x[1], x[0]))
                vocabs, _ = list(zip(*dicts[item]))
                dicts[item] = dict(zip(vocabs, range(1, len(vocabs) + 1)))
            dicts[item]['<unk>'] = 0
        return dicts

    @staticmethod
    def category_gen(dicts, column, val):
        if val not in dicts[column]:
            res = dicts[column]['<unk>']
        else:
            res = dicts[column][val]
        return res

    @staticmethod
    def continous_clip(feature_columns, continous_columns, data_file, percent=0.95):
        continous_clip = {}
        df = pd.read_csv(data_file, sep='\t', header=None, na_values=[r'\N'])
        df.columns = ['label'] + feature_columns
        for item in continous_columns:
            continous_clip[item] = df[item].dropna().quantile(percent)
        del df
        return continous_clip

    @staticmethod
    def continous_build(c_min, c_max, cliplist, features, table, continous_columns):
        for item in continous_columns:
            val = features[int(table[item]['index']) + 1]
            if val not in ['\\N', '', r'\N']:
                val = float(val)
                if val > cliplist[item]:
                    val = cliplist[item]
                c_min[item] = min(c_min[item], val)
                c_max[item] = max(c_max[item], val)

    @staticmethod
    def continous_gen(column, q_min, q_max, val):
        if val in ['\\N', '', r'\N']:
            return -100
        val = float(val)
        if q_min[column] == q_max[column]:
            return val
        else:
            return (val - q_min[column]) / (q_max[column] - q_min[column])

    @staticmethod
    def cdf_onehot(data_list, slice_num=20):
        thresholds = list()
        filtered_f_values = list(filter(lambda x: str(x) not in {'nan', 'None', '\\N', 'NaN', 'inf', r'\N'}, data_list))
        f_length = len(filtered_f_values)
        if f_length != 0:
            margin = 100.0 / slice_num
            percentiles = list()
            for i in range(slice_num - 1):
                percentiles.append(margin * (i + 1))
            thresholds_raw = np.percentile(np.array(filtered_f_values), percentiles, interpolation='lower')
            for i in range(thresholds_raw.size):
                if i == 1 or (thresholds_raw[i] - thresholds_raw[i - 1]) >= 0.000001:
                    thresholds.append(thresholds_raw[i])
        return thresholds