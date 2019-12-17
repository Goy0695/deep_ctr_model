import tensorflow as tf
import json
import os
import collections


class FFM:
    def __init__(self, config, params):
        self.model = None
        self.config = config
        global ffm_params
        ffm_params = params

    @staticmethod
    def input_fn(filenames, batch_size=32, num_epochs=1, perform_shuffle=False):
        def decode_libsvm(line):
            columns = tf.string_split([line], '\t')
            labels = tf.string_to_number(columns.values[0], out_type=tf.float32)
            splits = tf.string_split(columns.values[1:], ':')
            id_vals = tf.reshape(splits.values, splits.dense_shape)
            field_ids, feat_ids, feat_vals = tf.split(id_vals, num_or_size_splits=3, axis=1)
            field_ids = tf.string_to_number(field_ids, out_type=tf.int32)
            feat_ids = tf.string_to_number(feat_ids, out_type=tf.int32)
            feat_vals = tf.string_to_number(feat_vals, out_type=tf.float32)
            return {"field_ids": field_ids, "feat_ids": feat_ids, "feat_vals": feat_vals}, labels

        dataset = tf.data.TextLineDataset(filenames).map(decode_libsvm, num_parallel_calls=10).prefetch(500000)
        if perform_shuffle:
            dataset = dataset.shuffle(buffer_size=256)
        # epochs from blending together.
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.batch(batch_size)  # Batch size to use
        iterator = dataset.make_one_shot_iterator()
        batch_features, batch_labels = iterator.get_next()
        return batch_features, batch_labels

    @staticmethod
    def model_fn(features, labels, mode, params):
        column_size = params["column_size"]
        field_size = params["field_size"]
        feature_size = params["feature_size"]
        embedding_size = params["embedding_size"]
        l2_reg = params["l2_reg"]
        learning_rate = params["learning_rate"]
        batch_size = params["batch_size"]
        """
         column_size —— c
         feature_size —— n
         field_size ——f
         embedding_size ——k

        """
        # ----new weight and bias
        # shape [n]
        W1 = tf.get_variable(name='first_order_weight', shape=[feature_size],
                             initializer=tf.glorot_normal_initializer())
        # shape[n,f,k]
        W2 = tf.get_variable(name='second_order_weight', shape=[feature_size, field_size, embedding_size],
                             initializer=tf.glorot_normal_initializer())
        # ----get feature and label
        field_ids = features['field_ids']
        field_ids = tf.reshape(field_ids, shape=[-1, column_size])  # shape [None,c]
        feat_ids = features['feat_ids']
        feat_ids = tf.reshape(feat_ids, shape=[-1, column_size])  # shape [None,c]
        feat_vals = features['feat_vals']
        feat_vals = tf.reshape(feat_vals, shape=[-1, column_size])  # shape [None,c]
        field_ids_n = tf.reshape(
            [field_ids[i, j] + j * field_size + i * 6 for i in range(batch_size) for j in range(column_size)],
            [-1, column_size])  # shape [None,c]

        # ----build first order layer----
        with tf.variable_scope('first-order'):
            feat_wgts = tf.nn.embedding_lookup(W1, feat_ids)  # None * n
            y_w1 = tf.expand_dims(tf.reduce_sum(tf.multiply(feat_wgts, feat_vals), 1), 1)  # None * 1

        # ----build field interaction layer----
        with tf.variable_scope('second-order'):
            embeddings = tf.nn.embedding_lookup(tf.reshape(tf.nn.embedding_lookup(W2, feat_ids), [-1, embedding_size]),
                                                field_ids_n)  # None * c * K
            feat_vals = tf.reshape(feat_vals, shape=[-1, column_size, 1])  # None * c * 1
            # embeddings = f_embeddings[:,:,]
            y_w2 = tf.zeros_like(y_w1)
            for i in range(column_size - 1):
                for j in range(i + 1, column_size):
                    y_w2 = y_w2 + tf.expand_dims(
                        tf.reduce_sum(tf.multiply(embeddings[:, i, :], embeddings[:, j, :]), 1), 1) * feat_vals[:,
                                                                                                      i] * feat_vals[:,
                                                                                                           j]

        # ---- build output ----
        with tf.variable_scope("out"):
            y_bias = tf.get_variable(name='y_bias', shape=[1], initializer=tf.constant_initializer(0.0))
            y = y_bias + y_w1 + y_w2
            pred = tf.sigmoid(y)
        predictions = {"prob": pred}
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
        loss = l2_reg * tf.nn.l2_loss(W1) \
               + l2_reg * tf.nn.l2_loss(W2) \
               + tf.reduce_mean(tf.log(tf.ones_like(y, dtype=tf.float32) + tf.exp(-tf.multiply(labels, pred))))
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
            optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate, initial_accumulator_value=1.0)
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
        self.model = tf.estimator.Estimator(model_fn=FFM.model_fn, model_dir=model_dir, params=ffm_params,
                                            config=run_config)

    def train(self, tr_files, va_files):
        self.model.train(input_fn=lambda: FFM.input_fn(tr_files, num_epochs=ffm_params["num_epochs"],
                                                       batch_size=ffm_params["batch_size"]))

    def evaluate(self, va_files):
        self.model.evaluate(input_fn=lambda: FFM.input_fn(va_files, num_epochs=1, batch_size=ffm_params["batch_size"]))

    def train_and_evaluate(self, tr_files, va_files):
        evaluator = tf.estimator.experimental.InMemoryEvaluatorHook(
            estimator=self.model,
            input_fn=lambda: FFM.input_fn(va_files, num_epochs=1, batch_size=ffm_params["batch_size"]),
            every_n_iter=ffm_params["val_itrs"])
        self.model.train(
            input_fn=lambda: FFM.input_fn(tr_files, num_epochs=ffm_params["num_epochs"],
                                          batch_size=ffm_params["batch_size"]),
            hooks=[evaluator])

    def predict(self, te_files, isSave=False, numToSave=None):
        P_G = self.model.predict(input_fn=lambda: FFM.input_fn(te_files, num_epochs=1, batch_size=1),
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

    def export_model(self, output_path):
        filename = os.path.join(output_path, 'ffm.model')
        weight_dict = collections.defaultdict(dict)
        weight, bias = [], []
        weight.append(self.model.get_variable_value('first_order_weight').tolist())
        weight.append(self.model.get_variable_value('second_order_weight').tolist())
        bias.append(self.model.get_variable_value('y_bias').tolist())
        weight_dict['weight'] = weight
        weight_dict['bias'] = bias
        json_part = json.dumps(weight_dict)
        with open(filename, 'w') as f:
            f.write(json_part)