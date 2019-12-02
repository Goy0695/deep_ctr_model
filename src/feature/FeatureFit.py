from src.feature.FeatureInfo import FeatureInfo
from config.config import FLAGS

def transform(input_dir,output_dir):
    continous_features = range(1,FLAGS.continuous_field_size+1)
    categorial_features = range(FLAGS.continuous_field_size+1,FLAGS.field_size+1)
    FeatureInformation = FeatureInfo(continous_features,categorial_features)
    FeatureInformation.feamap(input_dir,output_dir)
    if FLAGS.modeltype in ["DeepFM","FM","FNN","LR","NFM","AFM","FFM"]:
        FeatureInformation.preprocess(input_dir,output_dir)
    else:
        FeatureInformation.preprocess_v2(input_dir,output_dir)

if __name__ == "__main__":
    transform(FLAGS.data_dir, FLAGS.data_dir)
