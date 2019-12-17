import tensorflow as tf
import os
import shutil
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score


class GBDT:

    def __init__(self, params):
        self.model = None
        self.num_trees = params.pop("num_trees")
        self.num_leaf = params["num_leaves"]
        self.isSave = params.pop("isSave")
        self.params = params

    def compile(self, model_dir=None):
        self.export_path = model_dir

    def train(self, tr_file, va_file):
        tr_file = tr_file[0]
        va_file = va_file[0]
        tra_data = pd.read_csv(tr_file, header=None, decimal=',', encoding='utf-8', keep_default_na=False)
        val_data = pd.read_csv(va_file, header=None, decimal=',', encoding='utf-8', keep_default_na=False)
        y_train = np.array(tra_data.pop(0))
        x_train = np.array(tra_data)
        y_val = np.array(val_data.pop(0))
        x_val = np.array(val_data)

        lgb_train = lgb.Dataset(x_train, y_train)
        lgb_val = lgb.Dataset(x_val, y_val)

        # Step1:训练gbdt模型
        self.model = lgb.train(self.params,
                               lgb_train,
                               num_boost_round=self.num_trees
                               )

        # Step2：计算所有样本在每棵树对应叶子节点的索引
        if self.isSave:
            train_indx = self.model.predict(x_train, pred_leaf=True).reshape(len(x_train),
                                                                             self.num_trees)  # shape:num_examples*num_trees
            val_indx = self.model.predict(x_val, pred_leaf=True).reshape(len(x_val),
                                                                         self.num_trees)  # shape:num_examples*num_trees
            df_train = pd.DataFrame(train_indx)
            df_val = pd.DataFrame(val_indx)
            df_train.insert(0, 'label', y_train.tolist())
            df_val.insert(0, 'label', y_val.tolist())
            if not os.path.exists('./data/gbdt_tmp'):
                os.mkdir('./data/gbdt_tmp/')

            df_train.to_csv('./data/gbdt_tmp/train.txt', sep='\t', header=0, index=False)
            df_val.to_csv('./data/gbdt_tmp/val.txt', sep='\t', header=0, index=False)

        # offset_indx = np.array([j*self.num_leaf for i in range(len(x_train)) for j in range(self.num_trees)]).reshape(len(x_train),self.num_trees)
        # input_indx = ori_indx + offset_indx
        # input_val = np.array([1]*len(x_train)*self.num_trees).reshape(len(x_train),self.num_trees)
        # labels = np.array(y_train).reshape(len(x_train),1)
        # self.output = (input_indx,input_val,labels)

    def evaluate(self, tr_file):
        tr_file = tr_file[0]
        tra_data = pd.read_csv(tr_file, header=None, decimal=',', encoding='utf-8')
        y = np.array(tra_data.pop(0))
        x = np.array(tra_data)
        y_pred = self.model.predict(x, pred_leaf=False)
        auc = roc_auc_score(y, y_pred)
        print('AUC on {} dataset is: {}'.format(tr_file, auc))

    def train_and_evaluate(self, tr_file, va_file):
        tr_file = tr_file[0]
        va_file = va_file[0]
        tra_data = pd.read_csv(tr_file, header=None, decimal=',', encoding='utf-8', keep_default_na=False)
        val_data = pd.read_csv(va_file, header=None, decimal=',', encoding='utf-8', keep_default_na=False)
        y_train = np.array(tra_data.pop(0))
        x_train = np.array(tra_data)
        y_val = np.array(val_data.pop(0))
        x_val = np.array(val_data)

        lgb_train = lgb.Dataset(x_train, y_train)
        lgb_val = lgb.Dataset(x_val, y_val)

        # Step1:训练gbdt模型
        self.model = lgb.train(self.params,
                               lgb_train,
                               num_boost_round=self.num_trees,
                               valid_sets=[lgb_train, lgb_val],
                               valid_names=['train', 'eval'],
                               verbose_eval=5
                               )

        # Step2：计算所有样本在每棵树对应叶子节点的索引
        if self.isSave:
            train_indx = self.model.predict(x_train, pred_leaf=True).reshape(len(x_train),
                                                                             self.num_trees)  # shape:num_examples*num_trees
            val_indx = self.model.predict(x_val, pred_leaf=True).reshape(len(x_val),
                                                                         self.num_trees)  # shape:num_examples*num_trees
            df_train = pd.DataFrame(train_indx)
            df_val = pd.DataFrame(val_indx)
            df_train.insert(0, 'label', y_train.tolist())
            df_val.insert(0, 'label', y_val.tolist())
            if not os.path.exists('./data/gbdt_tmp'):
                os.mkdir('./data/gbdt_tmp/')

            df_train.to_csv('./data/gbdt_tmp/train.txt', sep='\t', header=0, index=False)
            df_val.to_csv('./data/gbdt_tmp/val.txt', sep='\t', header=0, index=False)

    def predict(self, tr_file):
        return None

    def save(self, output_path):
        self.model.save_model(output_path)

