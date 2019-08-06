import tensorflow as tf
from src.base.model import basemodel
from Config import FLAGS
from src.feature.FeatureColumns import *
from src.feature.Feature import *

class WideDeep(basemodel):

    def __init__(self,params,config):
        self.params = params
        self.config = config
        self.model = None

    def set_feature_columns(self,input_dir):
        wide_columns = []
        deep_columns = []
        continous_features = range(1,FLAGS.continuous_field_size+1)
        categorial_features = range(FLAGS.continuous_field_size+1,FLAGS.field_size+1)

        dists = ContinuousFeatureGenerator(len(continous_features))
        dists.get_bounary(input_dir,continous_features)
        cdf_bounary = dists.cdf_bounary

        dicts = CategoryDictGenerator(len(categorial_features))
        dicts.build(input_dir, categorial_features, cutoff=0)

        #离散特征
        for idx in categorial_features:
            data_list = list(dicts.dicts[idx-categorial_features[0]].keys())
            data_list = ['-100' if item in ['nan', 'None', '\\N','NaN','inf'] else item for item in data_list]
            data_list = list(set(data_list))
            data_list.sort()
            column = FeatureColumns.one_hot(idx,data_list)
            wide_columns.append(column)
            deep_columns.append(FeatureColumns.column_embedding(column,FLAGS.embedding_size))

        for idx in continous_features:
            column = FeatureColumns.numeric(idx)
            deep_columns.append(column)
            wide_columns.append(FeatureColumns.column_bucket(column,cdf_bounary[idx-continous_features[0]]))
        self.params["deep_columns"] = deep_columns
        self.params["wide_columns"] = wide_columns

    @staticmethod
    def model_fn(features,labels,mode,params):
        ## ***************Wide part**************************
        layers = list(map(int, params["deep_layers"].split(',')))
        wide_net = []
        for value in params['wide_columns']:
            wide_net.append(value)
        lr_logit = tf.feature_column.linear_model(features=features,feature_columns=wide_net,units=1)
        ## ****************Deep part************************************
        deep_net = tf.feature_column.input_layer(features, params['deep_columns'])
        # hidden layer
        index = 0
        for unit in layers:
            index += 1
            layer_name = 'deep_layer_'+str(index)
            deep_net = tf.layers.dense(deep_net,units=unit,activation=tf.nn.relu,name=layer_name)
        deep_logit = tf.layers.dense(deep_net,units=1,activation=None,use_bias=False,name='deep_final')
        ## ***************Deep part + Wide part **************************
        logit = tf.add(deep_logit,lr_logit)
        ## ***************sigmoid*****************************************
        pred = tf.nn.sigmoid(logit)
        # call predict mode
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
            'prob': pred,
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit,labels=labels))
        auc = tf.contrib.metrics.streaming_auc(labels=labels,predictions=pred,name='auc_op')
        metrics = {'auc':auc}

        # define evaluate mode
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

        # define train mode
        assert mode == tf.estimator.ModeKeys.TRAIN
        optimizer = tf.train.AdagradOptimizer(learning_rate=params["learning_rate"])
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    def compile(self):
        run_config = self.config
        params = self.params
        self.model = tf.estimator.Estimator(model_fn= WideDeep.model_fn, params = params,config=run_config)


