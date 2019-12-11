import tensorflow as tf
import math
import json
import os
import collections


class LR:
    def __init__(self, config, params):
        self.model = None
        self.config = config
        global lr_params
        lr_params = params

    def input_fn(filenames, batch_size=32, num_epochs=1, perform_shuffle=False):
        print('Parsing', filenames)

        def decode_libsvm(line):
            # columns = tf.string_split([line], ' ')
            columns = tf.string_split([line], '\t')
            labels = tf.string_to_number(columns.values[0], out_type=tf.float32)
            splits = tf.string_split(columns.values[1:], ':')
            id_vals = tf.reshape(splits.values, splits.dense_shape)
            field_ids, feat_ids, feat_vals = tf.split(id_vals, num_or_size_splits=3, axis=1)
            field_ids = tf.string_to_number(field_ids, out_type=tf.int32)
            feat_ids = tf.string_to_number(feat_ids, out_type=tf.int32)
            feat_vals = tf.string_to_number(feat_vals, out_type=tf.float32)
            return {"feat_ids": feat_ids, "feat_vals": feat_vals}, labels

        """
        def decode_libsvm(line):
            columns = tf.string_split([line], ' ')
            labels = tf.string_to_number(columns.values[0], out_type=tf.float32)
            splits = tf.string_split(columns.values[1:], ':')
            id_vals = tf.reshape(splits.values,splits.dense_shape)
            feat_ids, feat_vals = tf.split(id_vals,num_or_size_splits=2,axis=1)
            feat_ids = tf.string_to_number(feat_ids, out_type=tf.int32)
            feat_vals = tf.string_to_number(feat_vals, out_type=tf.float32)
            return {"feat_ids": feat_ids, "feat_vals": feat_vals}, labels
        """
        dataset = tf.data.TextLineDataset(filenames).map(decode_libsvm, num_parallel_calls=10).prefetch(500000)
        if perform_shuffle:
            dataset = dataset.shuffle(buffer_size=256)
        # epochs from blending together.
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.batch(batch_size)  # Batch size to use
        iterator = dataset.make_one_shot_iterator()
        batch_features, batch_labels = iterator.get_next()
        return batch_features, batch_labels

    def model_fn(features, labels, mode, params):

        # labels = tf.cast(labels,dtype=tf.float32)
        l2_reg = params["l2_reg"]
        learning_rate = params["learning_rate"]
        field_size = params["field_size"]
        feature_size = params["feature_size"]

        # ------build W -------
        W = tf.get_variable(name='lr_weight', shape=[feature_size + 1], initializer=tf.glorot_normal_initializer())
        B = tf.get_variable(name='lr_bias', shape=[1], initializer=tf.constant_initializer(0.0))
        # ------build feature ------
        feat_ids = features['feat_ids']
        feat_ids = tf.reshape(feat_ids, shape=[-1, field_size])
        feat_vals = tf.cast(features['feat_vals'], dtype=tf.float32)
        feat_vals = tf.reshape(feat_vals, shape=[-1, field_size])  # None * num_trees * 1

        # ------build LR------
        with tf.variable_scope("LR"):
            f_w = tf.nn.embedding_lookup(W, feat_ids)  # None * num_trees * 1
            y_w = tf.reduce_sum(tf.multiply(f_w, feat_vals), 1)

        # -----build output-----
        with tf.variable_scope("out"):
            y_bias = B * tf.ones_like(y_w, dtype=tf.float32)  # None * 1
            y = y_w + y_bias
            pred = tf.sigmoid(y)

        predictions = {"y": y, "prob": pred}
        export_outputs = {
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(
                predictions)}
        # Provide an estimator spec for `ModeKeys.PREDICT`
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                export_outputs=export_outputs)

        # ------bulid loss------
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=labels)) + \
               l2_reg * tf.nn.l2_loss(W)

        # Provide an estimator spec for `ModeKeys.EVAL`
        eval_metric_ops = {
            "auc": tf.metrics.auc(labels, pred)
        }
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                loss=loss,
                eval_metric_ops=eval_metric_ops)

        # ------bulid optimizer------
        if params["optimizer"] == 'Adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
        elif params["optimizer"] == 'Adagrad':
            optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate, initial_accumulator_value=1e-8)
        elif params["optimizer"] == 'Momentum':
            optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.95)
        elif params["optimizer"] == 'ftrl':
            optimizer = tf.train.FtrlOptimizer(learning_rate)

        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

        # Provide an estimator spec for `ModeKeys.TRAIN` modes
        if mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                loss=loss,
                train_op=train_op)
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    def compile(self, model_dir=None):
        run_config = self.config
        self.model = tf.estimator.Estimator(model_fn=LR.model_fn, model_dir=model_dir, params=lr_params,
                                            config=run_config)

    def train(self, tr_files, va_files):
        self.model.train(input_fn=lambda: LR.input_fn(tr_files, num_epochs=lr_params["num_epochs"],
                                                      batch_size=lr_params["batch_size"]))

    def evaluate(self, va_files):
        self.model.evaluate(input_fn=lambda: LR.input_fn(va_files, num_epochs=1, batch_size=lr_params["batch_size"]))

    def predict(self, te_files, isSave=False, numToSave=None):
        P_G = self.model.predict(input_fn=lambda: LR.input_fn(te_files, num_epochs=1, batch_size=1),
                                 predict_keys="prob")
        if isSave:
            with open(te_files, 'r') as f1, open('sample.unitest', "w") as f2:
                if numToSave is None:
                    for sample in f1:
                        result = next(P_G)
                        pred = str(result['prob'])
                        f2.write('\t'.join([pred, sample]))
                else:
                    for i in range(numToSave):
                        sample = f1.readline()
                        result = next(P_G)
                        pred = str(result['prob'])
                        f2.write('\t'.join([pred, sample]))

    def single_predict(self, sample_str):
        '''
        Input:"1:0 2:0.012658 3:-0.011383 5:1 8:1 14:1 19:1 46:1 53:1 58:1 70:1"

        Output:Prediction of this sample

        '''

        def sigmoid(z):
            return 1 / (1 + math.exp(-z))

        feature_size = lr_params["feature_size"]
        weight = dict(
            zip([str(i) for i in range(1, feature_size + 1)], self.model.get_variable_value('lr_weight').tolist()))
        bias = self.model.get_variable_value('lr_bias')
        ids = [int(item.split(':')[0]) for item in sample_str.split(' ')]
        vals = [float(item.split(':')[1]) for item in sample_str.split(' ')]
        s = 0
        for ix, item in enumerate(ids):
            s += weight[str(item + 1)] * vals[ix]
        return sigmoid(s + bias)

    def export_model(self, output_path):
        filename = os.path.join(output_path, 'lr.model')
        weight_dict = collections.defaultdict(dict)
        weight, bias = [], []
        weight.append(self.model.get_variable_value('lr_weight').tolist())
        bias.append(self.model.get_variable_value('lr_bias').tolist())
        weight_dict['weight'] = weight
        weight_dict['bias'] = bias
        json_part = json.dumps(weight_dict)
        with open(filename, 'w') as f:
            f.write(json_part)

