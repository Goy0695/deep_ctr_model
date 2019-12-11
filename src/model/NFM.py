import tensorflow as tf


class NFM:

    def __init__(self, config, params):
        self.config = config
        self.model = None
        global nfm_params
        nfm_params = params

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
        """Bulid Model function f(x) for Estimator."""
        labels = tf.expand_dims(labels, 1)
        # ------hyperparameters----
        dropout = params["dropout"]
        field_size = params["field_size"]
        feature_size = params["feature_size"]
        embedding_size = params["embedding_size"]
        l2_reg = params["l2_reg"]
        learning_rate = params["learning_rate"]
        layers = list(map(int, params["deep_layers"].split(',')))

        # ------bulid weights------
        NFM_B = tf.get_variable(name='nfm_bias', shape=[1], initializer=tf.constant_initializer(0.0))
        NFM_W = tf.get_variable(name='nfm_w', shape=[feature_size], initializer=tf.glorot_normal_initializer())
        NFM_V = tf.get_variable(name='nfm_v', shape=[feature_size, embedding_size],
                                initializer=tf.glorot_normal_initializer())

        # ------build feaure-------
        feat_ids = features['feat_ids']
        feat_ids = tf.reshape(feat_ids, shape=[-1, field_size])
        feat_vals = features['feat_vals']
        feat_vals = tf.reshape(feat_vals, shape=[-1, field_size])

        # ------build f(x)------
        with tf.variable_scope("First-order"):
            feat_wgts = tf.nn.embedding_lookup(NFM_W, feat_ids)  # None * F
            y_w = tf.expand_dims(tf.reduce_sum(tf.multiply(feat_wgts, feat_vals), 1), 1)  # None * 1

        # ------build BI Interaction Layer -------
        with tf.variable_scope("Second-order"):
            embeddings = tf.nn.embedding_lookup(NFM_V, feat_ids)  # None * F * K
            feat_vals = tf.reshape(feat_vals, shape=[-1, field_size, 1])  # None * F * 1
            embeddings = tf.multiply(embeddings, feat_vals)  # vij*xi    # None * F * k
            sum_square = tf.square(tf.reduce_sum(embeddings, 1))  # None * k
            square_sum = tf.reduce_sum(tf.square(embeddings), 1)  # None * k
            bi_out = 0.5 * tf.subtract(sum_square, square_sum)  # None * k

        # ------add dropout layer
        if mode == tf.estimator.ModeKeys.TRAIN:
            deep_inputs = tf.nn.dropout(bi_out, keep_prob=dropout)
        else:
            deep_inputs = bi_out

        # ------build Deep Layers----------------
        with tf.variable_scope("Deep-part"):
            for i in range(len(layers)):
                deep_inputs = tf.contrib.layers.fully_connected(inputs=deep_inputs, num_outputs=layers[i],
                                                                activation_fn=tf.nn.relu,
                                                                weights_regularizer=tf.contrib.layers.l2_regularizer(
                                                                    l2_reg),
                                                                scope='mlp%d' % i)

        y_deep = tf.contrib.layers.fully_connected(inputs=deep_inputs, num_outputs=1, activation_fn=tf.identity,
                                                   weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
                                                   scope='deep_out')
        with tf.variable_scope("NFM-out"):
            y_bias = NFM_B * tf.ones_like(y_w, dtype=tf.float32)  # None * 1
            y = y_bias + y_w + y_deep
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
               l2_reg * tf.nn.l2_loss(NFM_W) + \
               l2_reg * tf.nn.l2_loss(NFM_V)

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
        self.model = tf.estimator.Estimator(model_fn=NFM.model_fn, model_dir=model_dir, params=nfm_params,
                                            config=run_config)

    def train(self, tr_files, va_files):
        self.model.train(input_fn=lambda: NFM.input_fn(tr_files, num_epochs=nfm_params["num_epochs"],
                                                       batch_size=nfm_params["batch_size"]))

    def evaluate(self, va_files):
        self.model.evaluate(input_fn=lambda: NFM.input_fn(va_files, num_epochs=1, batch_size=nfm_params["batch_size"]))

    def predict(self, te_files, isSave=False, numToSave=10):
        P_G = self.model.predict(input_fn=lambda: NFM.input_fn(te_files, num_epochs=1, batch_size=1),
                                 predict_keys="prob")
        if isSave:
            with open(te_files, 'r') as f1, open('sample.unitest', "w") as f2:
                for i in range(numToSave):
                    sample = f1.readline()
                    result = next(P_G)
                    pred = str(result['prob'])
                    f2.write('\t'.join([pred, sample]))


