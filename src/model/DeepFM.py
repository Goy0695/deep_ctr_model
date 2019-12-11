import tensorflow as tf


class DeepFM:

    def __init__(self, config, params):
        self.config = config
        self.model = None
        global deepfm_params
        deepfm_params = params

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

    def batch_norm_layer(x, train_phase, scope_bn, params):
        bn_train = tf.contrib.layers.batch_norm(x, decay=params['batch_norm_decay'], center=True, scale=True,
                                                updates_collections=None, is_training=True, reuse=None, scope=scope_bn)
        bn_infer = tf.contrib.layers.batch_norm(x, decay=params['batch_norm_decay'], center=True, scale=True,
                                                updates_collections=None, is_training=False, reuse=True, scope=scope_bn)
        z = tf.cond(tf.cast(train_phase, tf.bool), lambda: bn_train, lambda: bn_infer)
        return z

    def model_fn(features, labels, mode, params):
        """Bulid Model function f(x) for Estimator."""
        # ------hyperparameters----
        field_size = params["field_size"]
        feature_size = params["feature_size"]
        embedding_size = params["embedding_size"]
        l2_reg = params["l2_reg"]
        learning_rate = params["learning_rate"]
        layers = list(map(int, params["deep_layers"].split(',')))
        dropout = list(map(float, params["dropout"].split(',')))

        # ------bulid weights------
        FM_B = tf.get_variable(name='fm_bias', shape=[1], initializer=tf.constant_initializer(0.0))
        FM_W = tf.get_variable(name='fm_w', shape=[feature_size], initializer=tf.glorot_normal_initializer())
        FM_V = tf.get_variable(name='fm_v', shape=[feature_size, embedding_size],
                               initializer=tf.glorot_normal_initializer())

        # ------build feaure-------
        feat_ids = features['feat_ids']
        feat_ids = tf.reshape(feat_ids, shape=[-1, field_size])
        feat_vals = features['feat_vals']
        feat_vals = tf.reshape(feat_vals, shape=[-1, field_size])

        # ------build f(x)------
        with tf.variable_scope("First-order"):
            feat_wgts = tf.nn.embedding_lookup(FM_W, feat_ids)  # None * F * 1
            y_w = tf.reduce_sum(tf.multiply(feat_wgts, feat_vals), 1)

        with tf.variable_scope("Second-order"):
            embeddings = tf.nn.embedding_lookup(FM_V, feat_ids)  # None * F * K
            feat_vals = tf.reshape(feat_vals, shape=[-1, field_size, 1])  # None * F * 1
            embeddings = tf.multiply(embeddings, feat_vals)  # None * F * K
            sum_square = tf.square(tf.reduce_sum(embeddings, 1))  # None * F * K
            square_sum = tf.reduce_sum(tf.square(embeddings), 1)  # None * F * K
            y_v = 0.5 * tf.reduce_sum(tf.subtract(sum_square, square_sum), 1)  # None * 1

        with tf.variable_scope("Deep-part"):
            # if params["batch_norm"]:
            #    if mode == tf.estimator.ModeKeys.TRAIN:
            #        train_phase = True
            #    else:
            #        train_phase = False
            # else:
            #    normalizer_fn = None
            #    normalizer_params = None

            deep_inputs = tf.reshape(embeddings, shape=[-1, field_size * embedding_size])  # None * (F*K)
            for i in range(len(layers)):
                deep_inputs = tf.contrib.layers.fully_connected(inputs=deep_inputs, num_outputs=layers[i], \
                                                                weights_regularizer=tf.contrib.layers.l2_regularizer(
                                                                    l2_reg), scope='mlp%d' % i)
                # if params["batch_norm"]:
                #    deep_inputs = batch_norm_layer(deep_inputs, train_phase=train_phase, scope_bn='bn_%d' %i)   #放在RELU之后 https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md#bn----before-or-after-relu
                if mode == tf.estimator.ModeKeys.TRAIN:
                    deep_inputs = tf.nn.dropout(deep_inputs, keep_prob=dropout[
                        i])  # Apply Dropout after all BN layers and set dropout=0.8(drop_ratio=0.2)
            y_deep = tf.contrib.layers.fully_connected(inputs=deep_inputs, num_outputs=1, activation_fn=tf.identity, \
                                                       weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
                                                       scope='deep_out')
            y_d = tf.reshape(y_deep, shape=[-1])

        with tf.variable_scope("DeepFM-out"):
            y_bias = FM_B * tf.ones_like(y_d, dtype=tf.float32)  # None * 1
            y = y_bias + y_w + y_v + y_d
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
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=labels)) + \
               l2_reg * tf.nn.l2_loss(FM_W) + \
               l2_reg * tf.nn.l2_loss(FM_V)

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
        self.model = tf.estimator.Estimator(model_fn=DeepFM.model_fn, model_dir=model_dir, params=deepfm_params,
                                            config=run_config)

    def train(self, tr_files, va_file):
        self.model.train(input_fn=lambda: DeepFM.input_fn(tr_files, num_epochs=deepfm_params["num_epochs"],
                                                          batch_size=deepfm_params["batch_size"]))

    def evaluate(self, va_files):
        self.model.evaluate(
            input_fn=lambda: DeepFM.input_fn(va_files, num_epochs=1, batch_size=deepfm_params["batch_size"]))

    def predict(self, te_files, isSave=False, numToSave=None):
        P_G = self.model.predict(input_fn=lambda: DeepFM.input_fn(te_files, num_epochs=1, batch_size=1),
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
        return None

