import tensorflow as tf


class AFM:

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
        dataset = tf.data.TextLineDataset(filenames).map(decode_libsvm, num_parallel_calls=20).prefetch(500000)
        if perform_shuffle:
            dataset = dataset.shuffle(buffer_size=2048)
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
        field_size = params["field_size"]
        attention_size = params["attention_size"]
        feature_size = params["feature_size"]
        embedding_size = params["embedding_size"]
        l2_reg = params["l2_reg"]
        learning_rate = params["learning_rate"]

        # ------bulid weights------
        W = tf.get_variable(name='W', shape=[feature_size], initializer=tf.glorot_normal_initializer())
        AFM_V = tf.get_variable(name='AFM_V', shape=[feature_size, embedding_size],
                                initializer=tf.glorot_normal_initializer())
        AFM_W = tf.get_variable(name='AFM_W', shape=[embedding_size, attention_size],
                                initializer=tf.glorot_normal_initializer())
        AFM_B = tf.get_variable(name='AFM_B', shape=[AFM_W.shape[1], ], initializer=tf.constant_initializer(0.0))
        AFM_H = tf.get_variable(name='AFM_H', shape=[attention_size, 1], initializer=tf.glorot_normal_initializer())
        AFM_P = tf.get_variable(name='AFM_P', shape=[embedding_size, 1], initializer=tf.glorot_normal_initializer())

        # ------build feaure-------
        feat_ids = features['feat_ids']
        feat_ids = tf.reshape(feat_ids, shape=[-1, field_size])
        feat_vals = features['feat_vals']
        feat_vals = tf.reshape(feat_vals, shape=[-1, field_size])

        # ------build f(x)------
        with tf.variable_scope("First-order"):
            feat_wgts = tf.nn.embedding_lookup(W, feat_ids)  # None * F
            y_w = tf.expand_dims(tf.reduce_sum(tf.multiply(feat_wgts, feat_vals), 1), 1)  # None * 1

        # ------Pair-wise Interaction Layer-------
        with tf.variable_scope("Second-order"):
            embeddings = tf.nn.embedding_lookup(AFM_V, feat_ids)  # None * F * K
            feat_vals = tf.reshape(feat_vals, shape=[-1, field_size, 1])  # None * F * 1
            embeddings = tf.multiply(embeddings, feat_vals)

            ll = []
            for i in range(int(embeddings.shape[1])):
                for j in range(i + 1, int(embeddings.shape[1])):
                    ll.append(embeddings[:, i, :] * embeddings[:, j, :])

            # -------Attention-based Pooling layer---------
            interaction_size = int(field_size * (field_size - 1) / 2)
            vv = tf.reshape(tf.concat(ll, axis=1), shape=[-1, interaction_size, embedding_size])
            A_ = tf.tensordot(tf.nn.relu(tf.tensordot(vv, AFM_W, axes=1) + AFM_B), AFM_H, axes=1)
            A = tf.nn.softmax(A_, axis=0)
            A = tf.reduce_sum(A * vv, axis=1)
            y_afm = tf.tensordot(A, AFM_P, axes=1)

        with tf.variable_scope("AFM-out"):
            y_bias = tf.get_variable(name='y_bias', shape=[1], initializer=tf.constant_initializer(0.0))
            y = y_bias + y_w + y_afm
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
               l2_reg * tf.nn.l2_loss(AFM_W)

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
        self.model = tf.estimator.Estimator(model_fn=AFM.model_fn, model_dir=model_dir, params=nfm_params,
                                            config=run_config)

    def train(self, tr_files, va_files):
        self.model.train(input_fn=lambda: AFM.input_fn(tr_files, num_epochs=nfm_params["num_epochs"],
                                                       batch_size=nfm_params["batch_size"]))

    def evaluate(self, va_files):
        self.model.evaluate(input_fn=lambda: AFM.input_fn(va_files, num_epochs=1, batch_size=nfm_params["batch_size"]))

    def train_and_evaluate(self, tr_files, va_files):
        evaluator = tf.estimator.experimental.InMemoryEvaluatorHook(
            estimator=self.model,
            input_fn=lambda: AFM.input_fn(va_files, num_epochs=1, batch_size=afm_params["batch_size"]),
            every_n_iter=afm_params["val_itrs"])
        self.model.train(
            input_fn=lambda: AFM.input_fn(tr_files, num_epochs=afm_params["num_epochs"],
                                          batch_size=afm_params["batch_size"]),
            hooks=[evaluator])

    def predict(self, te_files, isSave=False, numToSave=10):
        P_G = self.model.predict(input_fn=lambda: AFM.input_fn(te_files, num_epochs=1, batch_size=1),
                                 predict_keys="prob")
        if isSave:
            with open(te_files, 'r') as f1, open('sample.unitest', "w") as f2:
                for i in range(numToSave):
                    sample = f1.readline()
                    result = next(P_G)
                    pred = str(result['prob'])
                    f2.write('\t'.join([pred, sample]))


