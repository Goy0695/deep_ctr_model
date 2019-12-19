import sys
import random
import pandas as pd
import os
from src.feature.Feature import Feature


class FeatureInfo:

    def __init__(self, con_method='minmax', cate_method='onehot'):
        self.con_method = con_method
        self.cate_method = cate_method
        self.fea = None
        self.feature_table = None
        self.feature_names = None
        self.category_feature_offset = None

    def feamap(self, input_dir, output_dir, tmp=False):
        """
        对连续型和类别型特征进行处理
        """

        fea = Feature(self.con_method, self.cate_method)
        print("Starting to load feature file!")
        if tmp == False:
            fea_dir = input_dir + '/feature'
            fea.load(fea_dir)
        else:
            fea.tmp_config()
        print("Starting to preprocess original data!")
        fea.build(input_dir, 150, 0.95)
        print("Starting to generate feature map!")
        output = open(output_dir + '/feature_map', 'w')
        for i, item in enumerate(fea.continous_columns):
            output.write("{0} {1}\n".format(item, i + 1))
        dict_sizes = [len(item) for key, item in fea.dicts.items()]
        category_feature_offset = [len(fea.continous_columns)]

        for i in range(1, len(fea.category_columns) + 1):
            offset = category_feature_offset[i - 1] + dict_sizes[i - 1]
            category_feature_offset.append(offset)
            for key, val in fea.dicts[fea.category_columns[i - 1]].items():
                output.write("{0} {1}\n".format(fea.category_columns[i - 1] + '|' + key,
                                                category_feature_offset[i - 1] + val + 1))
        self.category_feature_offset = category_feature_offset
        self.fea = fea
        self.feature_names = fea.feature_columns
        self.feature_table = fea.feature_table

    def libsvm_preprocess(self, input_dir, output_dir, thresold=0.9):
        # 90%的数据用于训练，10%的数据用于验证
        random.seed(2019)
        print("Process train data!")
        with open(output_dir + '/tr.libsvm', 'w') as out_train:
            with open(output_dir + '/va.libsvm', 'w') as out_valid:
                with open(input_dir + '/train.txt', 'r') as f:
                    for line in f:
                        features = line.rstrip('\n').split('\t')
                        feat_vals = []
                        indx = 0
                        for item in self.feature_names:
                            val = self.fea.gen(item, features[self.feature_table[item]['index'] + 1])
                            if self.feature_table[item]['type'] == 'q':
                                feat_vals.append(str(self.feature_table[item]['index']) \
                                                 + ':' + "{0:.6f}".format(val).rstrip('0').rstrip('.'))
                            else:
                                feat_vals.append(str(val + self.category_feature_offset[indx] + 1) + ':1')
                                indx = indx + 1
                        label = features[0]
                        if random.random() <= thresold:
                            out_train.write("{0}\t{1}\n".format(label, '\t'.join(feat_vals)))
                        else:
                            out_valid.write("{0}\t{1}\n".format(label, '\t'.join(feat_vals)))
        if os.path.exists(input_dir + '/test.txt'):
            print("Process test data!")
            with open(output_dir + '/te.libsvm', 'w') as out:
                with open(input_dir + '/test.txt', 'r') as f:
                    for line in f:
                        features = line.rstrip('\n').split('\t')
                        feat_vals = []
                        if len(self.feature_names) == len(features):
                            label = '0'
                            features.insert(0, '0')
                        else:
                            label = features[0]
                        indx = 0
                        for item in self.feature_names:
                            val = self.fea.gen(item, features[self.feature_table[item]['index'] + 1])
                            if self.feature_table[item]['type'] == 'q':
                                feat_vals.append(str(self.feature_table[item]['index']) \
                                                 + ':' + "{0:.6f}".format(val).rstrip('0').rstrip('.'))
                            else:
                                feat_vals.append(str(val + self.category_feature_offset[indx] + 1) + ':1')
                                indx = indx + 1
                        out.write("{0} {1}\n".format(label, ' '.join(feat_vals)))

    def ffm_preprocess_single(self, filename, phase='train'):
        print("Process {}!".format(filename))
        if phase == "train":
            out_file = os.path.join('/'.join('./data/gbdt_tmp/train.txt'.split('/')[:-1]), 'tr.libsvm')
        elif phase == 'val':
            out_file = os.path.join('/'.join('./data/gbdt_tmp/val.txt'.split('/')[:-1]), 'va.libsvm')
        else:
            out_file = os.path.join('/'.join('./data/gbdt_tmp/test.txt'.split('/')[:-1]), 'te.libsvm')
        with open(out_file, 'w') as out:
            with open(filename, 'r') as f:
                for line in f:
                    features = line.rstrip('\n').split('\t')
                    feat_vals = []
                    indx = 0
                    for item in self.feature_names:
                        field_id = str(self.feature_table[item]['field_id'])
                        val = self.fea.gen(item, features[self.feature_table[item]['index'] + 1])
                        if self.feature_table[item]['type'] == 'q':
                            feat_vals.append(field_id + ':' + str(self.feature_table[item]['index']) \
                                             + ':' + "{0:.6f}".format(val).rstrip('0').rstrip('.'))
                        else:
                            feat_vals.append(field_id + ':' + str(val + self.category_feature_offset[indx] + 1) + ':1')
                            indx = indx + 1
                    label = features[0]
                    out.write("{0}\t{1}\n".format(label, '\t'.join(feat_vals)))

    def csv_preprocess(self, input_dir, output_dir, thresold=0.9):
        random.seed(2019)
        # 90%的数据用于训练，10%的数据用于验证
        print("Process train data!")
        with open(output_dir + '/tr.csv', 'w') as out_train:
            with open(output_dir + '/va.csv', 'w') as out_valid:
                with open(input_dir + '/train.txt', 'r') as f:
                    for line in f:
                        features = line.rstrip('\n').split('\t')
                        feat_vals = [str(features[0])]
                        for item in self.feature_names:
                            if self.feature_table[item]['type'] == 'q':
                                val = self.fea.gen(item, features[self.feature_table[item]['index'] + 1])
                                feat_vals.append(str(val))
                            else:
                                val = features[self.feature_table[item]['index'] + 1]
                                if val not in ['\\N', r'\N', 'NAN', 'nan']:
                                    feat_vals.append(str(val))
                                else:
                                    feat_vals.append(str(-1))
                        if random.random() <= thresold:
                            out_train.write("{0}\n".format(','.join(feat_vals)))
                        else:
                            out_valid.write("{0}\n".format(','.join(feat_vals)))
        if os.path.exists(input_dir + '/test.txt'):
            print("Process test data!")
            with open(output_dir + '/te.csv', 'w') as out:
                with open(input_dir + '/test.txt', 'r') as f:
                    for line in f:
                        features = line.rstrip('\n').split('\t')
                        if len(self.feature_names) == len(features):
                            feat_vals = ['0']
                            features.insert(0, '0')
                        else:
                            feat_vals = [str(features[0])]
                        for item in self.feature_names:
                            if self.feature_table[item]['type'] == 'q':
                                val = self.fea.gen(item, features[self.feature_table[item]['index'] + 1])
                                feat_vals.append(str(val))
                            else:
                                val = features[self.feature_table[item]['index'] + 1]
                                if val not in ['\\N', r'\N', 'NAN', 'nan']:
                                    feat_vals.append(str(val))
                                else:
                                    feat_vals.append(str(-1))
                        out.write("{0}\n".format(','.join(feat_vals)))

    def ffm_preprocess(self, input_dir, output_dir, thresold=0.7):
        random.seed(2019)
        print("Process train data!")
        with open(output_dir + '/tr.libsvm', 'w') as out_train:
            with open(output_dir + '/va.libsvm', 'w') as out_valid:
                with open(input_dir + '/train.txt', 'r') as f:
                    for line in f:
                        features = line.rstrip('\n').split('\t')
                        feat_vals = []
                        indx = 0
                        for item in self.feature_names:
                            field_id = str(self.feature_table[item]['field_id'])
                            val = self.fea.gen(item, features[self.feature_table[item]['index'] + 1])
                            if self.feature_table[item]['type'] == 'q':
                                feat_vals.append(field_id + ':' + str(self.feature_table[item]['index']) \
                                                 + ':' + "{0:.6f}".format(val).rstrip('0').rstrip('.'))
                            else:
                                feat_vals.append(
                                    field_id + ':' + str(val + self.category_feature_offset[indx] + 1) + ':1')
                                indx = indx + 1
                        label = features[0]
                        if random.random() <= thresold:
                            out_train.write("{0}\t{1}\n".format(label, '\t'.join(feat_vals)))
                        else:
                            out_valid.write("{0}\t{1}\n".format(label, '\t'.join(feat_vals)))
        if os.path.exists(input_dir + '/test.txt'):
            print("Process test data!")
            with open(output_dir + '/te.libsvm', 'w') as out:
                with open(input_dir + '/test.txt', 'r') as f:
                    for line in f:
                        features = line.rstrip('\n').split('\t')
                        feat_vals = []
                        if len(self.feature_names) == len(features):
                            label = '0'
                            features.insert(0, '0')
                        else:
                            label = features[0]
                        indx = 0
                        for item in self.feature_names:
                            field_id = str(self.feature_table[item]['field_id'])
                            val = self.fea.gen(item, features[self.feature_table[item]['index'] + 1])
                            if self.feature_table[item]['type'] == 'q':
                                feat_vals.append(field_id + ':' + str(self.feature_table[item]['index']) \
                                                 + ':' + "{0:.6f}".format(val).rstrip('0').rstrip('.'))
                            else:
                                feat_vals.append(
                                    field_id + ':' + str(val + self.category_feature_offset[indx] + 1) + ':1')
                                indx = indx + 1
                        out.write("{0}\t{1}\n".format(label, '\t'.join(feat_vals)))