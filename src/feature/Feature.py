import sys
import random
import pandas as pd
import os
import numpy as np


class Feature():

    def __init__(self, con_method, cate_method):
        self.columns = ['index', 'field_id', 'type', 'method']
        self.feature_num = 0
        self.cliplist = []
        self.feature_columns = []
        self.category_columns = []
        self.continous_columns = []
        self.feature_table = {}
        self.q_min = {}
        self.q_max = {}
        self.q_max_abs = {}
        self.q_avg = {}
        self.q_sum = {}
        self.q_len = {}
        self.q_l2 = {}
        self.q_l1 = {}
        self.q_sd = {}
        self.dicts = {}
        self.dists = {}
        self.cdf_bounary = {}
        self.sample = None
        self.tmp = False
        self.con_method = con_method
        self.cate_method = cate_method
        print("Process Continous Feature With '{}' method".format(self.con_method))
        print("Process Category Feature With '{}' method".format(self.cate_method))

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
                    self.q_max_abs[f_name] = -sys.maxsize
                else:
                    self.category_columns.append(f_name)
                    self.dicts[f_name] = {}
                self.feature_columns.append(f_name)
                f_num = f_num + 1
            self.feature_num = f_num
            self.q_sum = dict(zip(self.continous_columns, [0] * len(self.continous_columns)))
            self.q_l2 = dict(zip(self.continous_columns, [0] * len(self.continous_columns)))
            self.q_l1 = dict(zip(self.continous_columns, [0] * len(self.continous_columns)))
            self.q_len = dict(zip(self.continous_columns, [1] * len(self.continous_columns)))
            self.q_avg = dict(zip(self.continous_columns, [0] * len(self.continous_columns)))
            self.q_sd = dict(zip(self.continous_columns, [0] * len(self.continous_columns)))

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

    def minmax(self, column, val):
        self.q_min[column] = min(self.q_min[column], val)
        self.q_max[column] = max(self.q_max[column], val)

    def absmax(self, column, val):
        self.q_max_abs[column] = max(self.q_max_abs[column], abs(val))

    def l1norm(self, column, val):
        self.q_l1[column] = self.q_l1[column] + abs(val)

    def l2norm(self, column, val):
        self.q_l2[column] = self.q_l2[column] + val ** 2

    def zscore(self, column, val):
        self.q_sd[column] = self.q_sd[column] + (val - self.q_avg[column]) ** 2

    def build(self, input_dir, cutoff, percent):
        datafile = input_dir + '/train.txt'
        if self.tmp == False:
            self.cliplist = Feature.continous_clip(self.feature_columns, self.continous_columns, datafile, percent)
        with open(datafile, 'r') as f1:
            for indx, line in enumerate(f1):
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
                    for item in self.continous_columns:
                        val = features[int(self.feature_table[item]['index']) + 1]
                        if val not in ['\\N', '', r'\N']:
                            val = float(val)
                            if val > self.cliplist[item]:
                                val = self.cliplist[item]
                            self.q_sum[item] = self.q_sum[item] + val
                            self.q_len[item] = self.q_len[item] + 1
            self.dicts = Feature.category_build(self.dicts, self.category_columns, 20)
        for k, v in self.q_sum.items():
            self.q_avg[k] = v / self.q_len[k]
        if self.tmp == False:
            with open(datafile, 'r') as f2:
                for indx, line in enumerate(f2):
                    features = line.rstrip('\n').split('\t')
                    for item in self.continous_columns:
                        val = features[int(self.feature_table[item]['index']) + 1]
                        if val not in ['\\N', '', r'\N']:
                            val = float(val)
                            if val > self.cliplist[item]:
                                val = self.cliplist[item]
                            if self.con_method == 'MinMax':
                                self.minmax(item, val)
                            elif self.con_method == 'AbsMax':
                                self.absmax(item, val)
                            elif self.con_method == 'L2Norm':
                                self.l2norm(item, val)
                            elif self.con_method == 'L1Norm':
                                self.l1norm(item, val)
                            else:
                                self.zscore(item, val)

    def gen(self, column, val):
        if column in self.category_columns:
            return Feature.category_gen(self.dicts, column, val)
        else:
            if self.con_method == 'MinMax':
                return Feature.minmax_gen(column, self.q_min, self.q_max, self.q_avg, val)
            elif self.con_method == 'AbsMax':
                return Feature.absmax_gen(column, self.q_max_abs, self.q_avg, val)
            elif self.con_method == 'L2Norm':
                return Feature.l2norm_gen(column, self.q_l2, self.q_avg, val)
            elif self.con_method == 'L1Norm':
                return Feature.l1norm_gen(column, self.q_l1, self.q_avg, val)
            else:
                return Feature.zscore_gen(column, self.q_sd, self.q_len, self.q_avg, val)

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
                # dicts[item] = sorted(dicts[item], key=lambda x: (-x[1], x[0]))
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
    # min-max 归一化
    def minmax_gen(column, q_min, q_max, q_avg, val):
        if val in ['\\N', '', r'\N']:
            val = q_avg[column]
        else:
            val = float(val)
        if q_min[column] == q_max[column]:
            return val
        else:
            return (val - q_min[column]) / (q_max[column] - q_min[column])

    @staticmethod
    # abs-max 归一化
    def absmax_gen(column, q_max_abs, q_avg, val):
        if val in ['\\N', '', r'\N']:
            val = q_avg[column]
        else:
            val = float(val)
        if q_max_abs[column] == 0:
            return val
        else:
            return val / q_max_abs[column]

    @staticmethod
    # 标准化-zscore
    def zscore_gen(column, q_sd, q_len, q_avg, val):
        if val in ['\\N', '', r'\N']:
            val = q_avg[column]
        else:
            val = float(val)
        if q_sd[column] == 0:
            return val
        else:
            return (val - q_avg[column]) / np.sqrt(q_sd[column] / q_len[column])

    @staticmethod
    # l2-norm
    def l2norm_gen(column, q_l2, q_avg, val):
        if val in ['\\N', '', r'\N']:
            val = q_avg[column]
        else:
            val = float(val)
        if q_l2[column] == 0:
            return val
        else:
            return val / np.sqrt(q_l2[column])

    @staticmethod
    # l1-norm
    def l1norm_gen(column, q_l1, q_avg, val):
        if val in ['\\N', '', r'\N']:
            val = q_avg[column]
        else:
            val = float(val)
        if q_l1[column] == 0:
            return val
        else:
            return val / q_l1[column]

    @staticmethod
    # 等间距分桶
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