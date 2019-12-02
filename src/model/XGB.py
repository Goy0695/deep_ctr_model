import xgboost as xgb
import pandas as pd
import numpy as np
import gc
from sklearn.metrics import roc_auc_score

class XGB:

    def __init__(self,params):
        '''
        Input:
        :param tr_file:
        :param va_file:
        :param te_file:
        '''
        # python api 提供的xgboost简直就是鸡肋，fit函数不支持Dmatrix输入
        #self.tra_data = xgb.DMatrix(tr_file+'?format=csv&label_column=0')
        #self.val_data = xgb.DMatrix(va_file+'?format=csv&label_column=0')
        #self.test_dat = xgb.DMatrix(te_file+'?format=csv&label_column=0')
        self.model = None
        self.export_path = None
        global xgb_params
        xgb_params = params

    def compile(self,model_dir=None):
        self.export_path = model_dir
        self.model =  xgb.XGBModel(**xgb_params)

    def train(self,tr_file,va_file):
        tr_file = tr_file[0]
        va_file = va_file[0]
        tra_data = pd.read_csv(tr_file,header=None,decimal=',',encoding='utf-8')
        val_data = pd.read_csv(va_file, header=None, decimal=',', encoding='utf-8')
        y_train = np.array(tra_data.pop(0))
        x_train = np.array(tra_data)
        y_val = np.array(val_data.pop(0))
        x_val = np.array(val_data)
        self.model.fit(
                        x_train,
                        y_train,
                        eval_set=[(x_val,y_val)],
                        eval_metric=["auc","logloss"],
                        early_stopping_rounds=10,
                        verbose=10
                        )

    def evaluate(self,tr_file):
        tr_file = tr_file[0]
        data = pd.read_csv(tr_file, header=None, decimal=',', encoding='utf-8')
        y = np.array(data.pop(0))
        x = np.array(data)
        y_pred = self.model.predict(data=x,output_margin=False)
        auc = roc_auc_score(y, y_pred)
        print('AUC on {} dataset is: {}'.format(tr_file,auc))

    def predict(self,tr_file,isSave=False):
        tr_file = tr_file[0]
        data = pd.read_csv(tr_file, header=None, decimal=',', encoding='utf-8')
        y = np.array(data.pop(0))
        x = np.array(data)
        y_pred = self.model.predict(data=x,output_margin=False)
        return y_pred
        if isSave:
            y_pred.tofile(phase+"_pred.txt")

    def get_leaf_indx(self,tr_file):
        '''
        Parameters
        ----------
        Phase : decide to process which kind of dataset

        Returns
        -------
        X_leaves : array_like, shape=[n_samples, n_trees]
        '''
        tr_file = tr_file[0]
        data = pd.read_csv(tr_file, header=None, decimal=',', encoding='utf-8')
        y = np.array(data.pop(0))
        x = np.array(data)
        return self.model.apply(x)

    def save(self,output_path):
        self.model.save_model(self.export_path)
