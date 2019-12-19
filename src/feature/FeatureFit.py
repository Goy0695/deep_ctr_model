from src.feature.FeatureInfo import FeatureInfo
from config.config import FLAGS


def transform(input_dir, output_dir, con_method='minmax', cate_method='onehot'):
    print("The model type is {}!".format(FLAGS.modeltype))
    FeatureInformation = FeatureInfo(con_method, cate_method)
    FeatureInformation.feamap(input_dir, output_dir)
    print("Starting to split train data!")
    if FLAGS.modeltype in ["DeepFM", "FM", "FNN", "LR", "NFM", "AFM", "FFM"]:
        FeatureInformation.ffm_preprocess(input_dir, output_dir)
    else:
        FeatureInformation.csv_preprocess(input_dir, output_dir)


if __name__ == "__main__":
    transform(FLAGS.data_dir, FLAGS.data_dir)
