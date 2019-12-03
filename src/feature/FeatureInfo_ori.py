import random
from src.feature.Feature_ori import CategoryDictGenerator, ContinuousFeatureGenerator
import os


class FeatureInfo:

    def __init__(self, confeaindx, catefeaindx):
        self.continous_features = confeaindx
        self.categorial_features = catefeaindx
        self.dists = None
        self.dicts = None
        self.categorial_feature_offset = None

    def feamap(self, input_dir, output_dir):
        """
        对连续型和类别型特征进行处理
        """
        dists = ContinuousFeatureGenerator(len(self.continous_features))
        dists.clip(input_dir, self.continous_features, 0.95)
        dists.build(input_dir, self.continous_features)

        dicts = CategoryDictGenerator(len(self.categorial_features))
        dicts.build(input_dir, self.categorial_features, cutoff=150)

        output = open(output_dir + '/feature_map', 'w')
        for i in self.continous_features:
            output.write("{0} {1}\n".format('I' + str(i), i))
        dict_sizes = dicts.dicts_sizes()
        categorial_feature_offset = [dists.num_feature]
        for i in range(1, len(self.categorial_features) + 1):
            offset = categorial_feature_offset[i - 1] + dict_sizes[i - 1]
            categorial_feature_offset.append(offset)
            for key, val in dicts.dicts[i - 1].items():
                output.write("{0} {1}\n".format('C' + str(i) + '|' + key, categorial_feature_offset[i - 1] + val + 1))
        self.categorial_feature_offset = categorial_feature_offset
        self.dists = dists
        self.dicts = dicts

    def preprocess(self, input_dir, output_dir, thresold=0.9):
        # 90%的数据用于训练，10%的数据用于验证
        random.seed(2019)
        """
        print("Process train data!")
        with open(output_dir + '/tr.libsvm', 'w') as out_train:
            with open(output_dir + '/va.libsvm', 'w') as out_valid:
                with open(input_dir + '/train.txt', 'r') as f:
                    for line in f:
                        features = line.rstrip('\n').split(',')
                        feat_vals = []
                        for i in range(0, len(self.continous_features)):
                            val = self.dists.gen(i, features[self.continous_features[i]])
                            feat_vals.append(str(self.continous_features[i]) + ':' + "{0:.6f}".format(val).rstrip('0').rstrip('.'))

                        for i in range(0, len(self.categorial_features)):
                            val = self.dicts.gen(i, features[self.categorial_features[i]]) + self.categorial_feature_offset[i] + 1
                            feat_vals.append(str(val) + ':1')
                        label = features[0]
                        if random.random() <=thresold:
                            out_train.write("{0} {1}\n".format(label, ' '.join(feat_vals)))
                        else:
                            out_valid.write("{0} {1}\n".format(label, ' '.join(feat_vals)))
                            """
        print("Process test data!")
        with open(output_dir + '/te.libsvm', 'w') as out:
            with open(input_dir + '/test.txt', 'r') as f:
                for line in f:
                    features = line.rstrip('\n').split(',')
                    feat_vals = []
                    if len(list(self.continous_features) + list(self.categorial_features)) == len(features):
                        label = '0'
                        features.insert(0, '0')
                    else:
                        label = features[0]
                    for i in range(0, len(self.continous_features)):
                        val = self.dists.gen(i, features[self.continous_features[i]])
                        feat_vals.append(
                            str(self.continous_features[i]) + ':' + "{0:.6f}".format(val).rstrip('0').rstrip('.'))
                    for i in range(0, len(self.categorial_features)):
                        val = self.dicts.gen(i, features[self.categorial_features[i]]) + self.categorial_feature_offset[
                            i] + 1
                        feat_vals.append(str(val) + ':1')
                    out.write("{0} {1}\n".format(label, ' '.join(feat_vals)))

    def preprocess_single(self, filename, phase='train'):
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
                    features = line.rstrip('\n').split(',')
                    feat_vals = []
                    for i in range(0, len(self.continous_features)):
                        val = self.dists.gen(i, features[self.continous_features[i]])
                        feat_vals.append(
                            str(self.continous_features[i]) + ':' + "{0:.6f}".format(val).rstrip('0').rstrip('.'))
                    for i in range(0, len(self.categorial_features)):
                        val = self.dicts.gen(i, features[self.categorial_features[i]]) + self.categorial_feature_offset[
                            i] + 1
                        feat_vals.append(str(val) + ':1')
                    label = features[0]
                    out.write("{0} {1}\n".format(label, ' '.join(feat_vals)))

    def preprocess_v2(self, input_dir, output_dir):
        random.seed(2019)
        # 90%的数据用于训练，10%的数据用于验证
        print("Process train data!")
        with open(output_dir + '/tr.csv', 'w') as out_train:
            with open(output_dir + '/va.csv', 'w') as out_valid:
                with open(input_dir + '/train.txt', 'r') as f:
                    for line in f:
                        features = line.rstrip('\n').split(',')
                        feat_vals = [str(features[0])]
                        for i in range(0, len(self.continous_features)):
                            val = self.dists.gen(i, features[self.continous_features[i]])
                            feat_vals.append(str(val))
                        for i in range(0, len(self.categorial_features)):
                            val = features[self.categorial_features[i]]
                            feat_vals.append(str(val))
                        if random.randint(0, 9999) % 10 != 0:
                            out_train.write("{0}\n".format(','.join(feat_vals)))
                        else:
                            out_valid.write("{0}\n".format(','.join(feat_vals)))
        print("Process test data!")
        with open(output_dir + '/te.csv', 'w') as out:
            with open(input_dir + '/test.txt', 'r') as f:
                for line in f:
                    features = line.rstrip('\n').split(',')
                    if len(list(self.continous_features) + list(self.categorial_features)) == len(features):
                        feat_vals = ['0']
                        features.insert(0, '0')
                    else:
                        feat_vals = [str(features[0])]
                    for i in range(0, len(self.continous_features)):
                        val = self.dists.gen(i, features[self.continous_features[i]])
                        feat_vals.append(str(val))
                    for i in range(0, len(self.categorial_features)):
                        val = features[self.categorial_features[i]]
                        feat_vals.append(str(val))
                    out.write("{0}\n".format(','.join(feat_vals)))


