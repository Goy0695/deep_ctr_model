from src.model.GBDT import GBDT
from src.model.FM import FM
import re
from src.feature.FeatureInfo import FeatureInfo


class GBDT_FM:

    def __init__(self, config, params1, params2):
        self.GBDT = GBDT(params1)
        self.FM = FM(config, params2)
        # self.continous_features = range(1,params2['continuous_field_size']+1)
        # self.categorial_features = range(params2['continuous_field_size']+1,params2['field_size']+1)

    def compile(self, model_dir=None):
        self.FM.compile(model_dir)

    def train(self, tr_file, va_file):
        print("*************Start to train gbdt model******************")
        self.GBDT.train(tr_file, va_file)
        print("*************Preprocess data for fm model******************")
        FeatureInformation = FeatureInfo()
        FeatureInformation.feamap('./data/gbdt_tmp/', './data/gbdt_tmp/', tmp=True)
        FeatureInformation.ffm_preprocess_single('./data/gbdt_tmp/train.txt', phase='train')
        FeatureInformation.ffm_preprocess_single('./data/gbdt_tmp/val.txt', phase='val')
        tr_gbdt_files, va_gbdt_files = ['./data/gbdt_tmp/tr.libsvm'], ['./data/gbdt_tmp/va.libsvm']
        print("*************Start to train fm model******************")
        self.FM.train(tr_gbdt_files, va_gbdt_files)

    def evaluate(self, tr_file):
        print("*************Start to evaluate fm model******************")
        if re.findall('/([a-z]+).csv', tr_file[0])[0] == 'tr':
            self.FM.evaluate(['./data/gbdt_tmp/tr.libsvm'])
        elif re.findall('/([a-z]+).csv', tr_file[0])[0] == 'va':
            self.FM.evaluate(['./data/gbdt_tmp/va.libsvm'])
        else:
            print("Data type is not support yet! Please input tr.libsvm or va.libsvm!")


