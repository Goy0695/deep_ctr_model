import collections
import pandas as pd
import sys
import numpy as np

class CategoryDictGenerator:
    """
    类别型特征编码字典
    """

    def __init__(self, num_feature):
        self.dicts = []
        self.num_feature = num_feature

        for i in range(0, num_feature):
            self.dicts.append(collections.defaultdict(int))

    def build(self, input_dir, categorial_features, cutoff=0):
        datafile = input_dir + '/criteo_train.txt'
        with open(datafile, 'r') as f:
            for line in f:
                features = line.rstrip('\n').split(',')
                for i in range(0, self.num_feature):
                    if features[categorial_features[i]] != '':
                        self.dicts[i][features[categorial_features[i]]] += 1
        for i in range(0, self.num_feature):
            self.dicts[i] = filter(lambda x: x[1] >= cutoff, self.dicts[i].items())
            self.dicts[i] = sorted(self.dicts[i], key=lambda x: (-x[1], x[0]))
            vocabs, _ = list(zip(*self.dicts[i]))
            self.dicts[i] = dict(zip(vocabs, range(1, len(vocabs) + 1)))
            self.dicts[i]['<unk>'] = 0

    def gen(self, idx, key):
        if key not in self.dicts[idx]:
            res = self.dicts[idx]['<unk>']
        else:
            res = self.dicts[idx][key]
        return res

    def dicts_sizes(self):
        return list(map(len, self.dicts))

class ContinuousFeatureGenerator:
    """
    对连续值特征做最大最小值normalization
    """

    def __init__(self, num_feature):
        self.num_feature = num_feature
        self.min = [sys.maxsize] * num_feature
        self.max = [-sys.maxsize] * num_feature
        self.cliplist = []
        self.cdf_bounary = []

    def clip(self,input_dir,continous_features,percent=0.95):
        input_path = input_dir + '/criteo_train.txt'
        continous_clip = []
        df = pd.read_csv(input_path,sep=',',header=None)
        for i in continous_features:
            continous_clip.append(df[i].dropna().quantile(percent))
        del df
        self.cliplist = continous_clip

    def get_bounary(self,input_dir,continous_features):
        input_path = input_dir + '/criteo_train.txt'
        cdf_table = {}
        df = pd.read_csv(input_path,sep=',',header=None)
        for i in continous_features:
            feature_column = df[i].tolist()
            self.cdf_bounary.append(ContinuousFeatureGenerator.cdf_onehot(feature_column))
        del df

    def build(self, input_dir, continous_features):
        datafile = input_dir + '/criteo_train.txt'
        with open(datafile, 'r') as f:
            for line in f:
                features = line.rstrip('\n').split(',')
                for i in range(0, self.num_feature):
                    val = features[continous_features[i]]
                    if val != '':
                        val = int(val)
                        if val > self.cliplist[i]:
                            val = self.cliplist[i]
                        self.min[i] = min(self.min[i], val)
                        self.max[i] = max(self.max[i], val)

    def gen(self, idx, val):
        if val == '':
            return -100
        val = float(val)
        return (val - self.min[idx]) / (self.max[idx] - self.min[idx])

    @staticmethod
    def cdf_onehot(data_list,slice_num = 20):
        thresholds = list()
        filtered_f_values = list(filter(lambda x: str(x) not in {'nan', 'None', '\\N','NaN','inf'}, data_list))
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