from src.feature.FeatureColumns import *
from src.feature.Feature import *


class WideDeep:

    def __init__(self, config, params):
        self.config = config
        self.model = None
        global widedeep_params
        widedeep_params = params

    def input_fn(filenames, batch_size=32, num_epochs=1, perform_shuffle=False):
        print('Parsing', filenames)

        def decode_csv(line):
            continus_sample = [[0.0] for i in range(widedeep_params["continuous_field_size"])]
            category_sample = [["0.0"] for i in range(widedeep_params["category_field_size"])]
            sample = [[0.0]] + continus_sample + category_sample
            item = tf.decode_csv(line, sample)
            feature = item[1:]
            label = tf.expand_dims(item[0], -1)
            indx = [str(item) for item in range(1, widedeep_params["field_size"] + 1)]
            return dict(zip(indx, feature)), label

        dataset = tf.data.TextLineDataset(filenames).map(decode_csv, num_parallel_calls=10).prefetch(500000)
        if perform_shuffle:
            dataset = dataset.shuffle(buffer_size=256)
        # epochs from blending together.
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.batch(batch_size)  # Batch size to use
        iterator = dataset.make_one_shot_iterator()
        batch_features, batch_labels = iterator.get_next()
        return batch_features, batch_labels

    def set_feature_columns(self, input_dir):
        wide_columns = []
        deep_columns = []
        continous_features = range(1, widedeep_params["continuous_field_size"] + 1)
        categorial_features = range(widedeep_params["continuous_field_size"] + 1, widedeep_params["field_size"] + 1)

        dists = ContinuousFeatureGenerator(len(continous_features))
        dists.get_bounary(input_dir, continous_features)
        cdf_bounary = dists.cdf_bounary

        dicts = CategoryDictGenerator(len(categorial_features))
        dicts.build(input_dir, categorial_features, cutoff=0)

        # 离散特征
        for idx in categorial_features:
            data_list = list(dicts.dicts[idx - categorial_features[0]].keys())
            data_list = ['-100' if item in ['nan', 'None', '\\N', 'NaN', 'inf'] else item for item in data_list]
            data_list = list(set(data_list))
            data_list.sort()
            column = FeatureColumns.one_hot(idx, data_list)
            wide_columns.append(column)
            deep_columns.append(FeatureColumns.column_embedding(column, widedeep_params["embedding_size"]))

        if widedeep_params["is_cross"]:
            cross_features = widedeep_params["cross_columns"].split(',')
            for item in cross_features:
                cross_indx = item.split(':')
                indx_list = cross_indx[0].split("&")
                bucket_size = int(cross_indx[1])
                column = FeatureColumns.column_cross(indx_list, bucket_size)
                wide_columns.append(column)
                deep_columns.append(FeatureColumns.column_embedding(column, widedeep_params["embedding_size"]))

        for idx in continous_features:
            column = FeatureColumns.numeric(idx)
            deep_columns.append(column)
            wide_columns.append(FeatureColumns.column_bucket(column, cdf_bounary[idx - continous_features[0]]))
        widedeep_params["deep_columns"] = deep_columns
        widedeep_params["wide_columns"] = wide_columns

    @staticmethod
    def model_fn(features, labels, mode, params):
        ## ***************Wide part**************************
        layers = list(map(int, params["deep_layers"].split(',')))
        wide_net = []
        for value in params['wide_columns']:
            wide_net.append(value)
        lr_logit = tf.feature_column.linear_model(features=features, feature_columns=wide_net, units=1)
        ## ****************Deep part************************************
        deep_net = tf.feature_column.input_layer(features, params['deep_columns'])
        # hidden layer
        index = 0
        for unit in layers:
            index += 1
            layer_name = 'deep_layer_' + str(index)
            deep_net = tf.layers.dense(deep_net, units=unit, activation=tf.nn.relu, name=layer_name)
        deep_logit = tf.layers.dense(deep_net, units=1, activation=None, use_bias=False, name='deep_final')
        ## ***************Deep part + Wide part **************************
        logit = tf.add(deep_logit, lr_logit)
        ## ***************sigmoid*****************************************
        pred = tf.nn.sigmoid(logit)
        # call predict mode
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                'prob': pred,
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit, labels=labels))
        auc = tf.contrib.metrics.streaming_auc(labels=labels, predictions=pred, name='auc_op')
        metrics = {'auc': auc}

        # define evaluate mode
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

        # define train mode
        assert mode == tf.estimator.ModeKeys.TRAIN
        optimizer = tf.train.AdagradOptimizer(learning_rate=params["learning_rate"])
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    def compile(self, model_dir=None):
        run_config = self.config
        self.model = tf.estimator.Estimator(model_fn=WideDeep.model_fn, model_dir=model_dir, params=widedeep_params,
                                            config=run_config)

    def train(self, tr_files, va_files):
        self.model.train(input_fn=lambda: WideDeep.input_fn(tr_files, num_epochs=widedeep_params["num_epochs"],
                                                            batch_size=widedeep_params["batch_size"]))

    def evaluate(self, va_files):
        self.model.evaluate(
            input_fn=lambda: WideDeep.input_fn(va_files, num_epochs=1, batch_size=widedeep_params["batch_size"]))

    def predict(self, te_files, isSave=False):
        P_G = self.model.predict(
            input_fn=lambda: WideDeep.input_fn(te_files, num_epochs=1, batch_size=widedeep_params["batch_size"]),
            predict_keys="prob")
        return P_G



