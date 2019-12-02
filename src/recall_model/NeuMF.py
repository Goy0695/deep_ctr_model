import numpy as np
import tensorflow as tf


class NeuMF:

    def __init__(self, config, params, infile, user_cnt=None, item_cnt=None):
        self.config = config
        self.model = None
        self.get_info(infile)
        global mf_params
        mf_params = params
        mf_params['user_cnt'] = self.user_cnt
        mf_params['item_cnt'] = self.item_cnt
        mf_params['mu'] = self.mu

    def get_info(self, filenames):
        import numpy as np
        rate = []
        user = set()
        item = set()
        with open(filenames, 'r') as f:
            for line in f:
                words = line.replace('\r\n', '').replace('\n', '').split('\t')
                user.add(int(words[0]))
                item.add(int(words[1]))
                rate.append(float(words[2]))
        self.mu = np.mean(rate)
        self.user_cnt = len(user)
        self.item_cnt = len(item)

    def input_fn(filenames, batch_size=64, num_epochs=1, perform_shuffle=False):
        print('Parsing', filenames)

        def decode_sparse(line):
            columns = tf.string_split([line], '\t')
            user = tf.string_to_number(columns.values[0], out_type=tf.int32)
            item = tf.string_to_number(columns.values[1], out_type=tf.int32)
            rate = tf.string_to_number(columns.values[2], out_type=tf.float32)
            return {"user_id": user, "item_id": item}, rate

        dataset = tf.data.TextLineDataset(filenames).map(decode_sparse, num_parallel_calls=10).prefetch(10000)
        if perform_shuffle:
            dataset = dataset.shuffle(buffer_size=256)
        # epochs from blending together.
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.batch(batch_size)  # Batch size to use
        iterator = dataset.make_one_shot_iterator()
        batch_features, batch_labels = iterator.get_next()
        return batch_features, batch_labels

    def add_layer(inputs, num_outputs, activation=None, sub_name=None):
        in_units = inputs.shape[1]
        out_units = num_outputs
        weights = tf.get_variable(name=sub_name + "_weight", shape=[in_units, out_units],
                                  initializer=tf.glorot_normal_initializer(), dtype=tf.float32)
        bias = tf.get_variable(name=sub_name + "_bias", shape=[out_units, ], initializer=tf.zeros_initializer(),
                               dtype=tf.float32)
        Z = tf.add(tf.matmul(inputs, weights), bias)
        if activation:
            return activation(Z)
        return Z

    def model_fn(features, labels, mode, params):
        """Bulid Model function f(x) for Estimator."""
        user_size = params["user_cnt"]
        item_size = params["item_cnt"]
        layers = params["layers"]
        l2_reg = params["l2_reg"]
        learning_rate = params["learning_rate"]

        # ------bulid weights------
        W_mlp_user = tf.get_variable(name='user_mlp_embedding', shape=[user_size, layers[0]],
                                     initializer=tf.glorot_normal_initializer())
        W_mlp_item = tf.get_variable(name='item_mlp_embedding', shape=[item_size, layers[0]],
                                     initializer=tf.glorot_normal_initializer())
        W_mf_user = tf.get_variable(name='user_mf_embedding', shape=[user_size, layers[0]],
                                    initializer=tf.glorot_normal_initializer())
        W_mf_item = tf.get_variable(name='item_mf_embedding', shape=[item_size, layers[0]],
                                    initializer=tf.glorot_normal_initializer())

        # ------build (user,item) pair-------
        user_ids = features['user_id']
        item_ids = features['item_id']

        # ------build input for MF-----------
        user_mf_embedding = tf.nn.embedding_lookup(W_mf_user, user_ids)  # None*(k+1)
        item_mf_embedding = tf.nn.embedding_lookup(W_mf_item, item_ids)  # None*(k+1)
        mf_inputs = tf.multiply(user_mf_embedding, item_mf_embedding)

        # ------build input for DNN----------
        user_mlp_embedding = tf.nn.embedding_lookup(W_mlp_user, user_ids)  # None*(k+1)
        item_mlp_embedding = tf.nn.embedding_lookup(W_mlp_item, item_ids)  # None*(k+1)
        deep_inputs = tf.concat([user_mlp_embedding, item_mlp_embedding], 1)

        with tf.variable_scope("Deep-part"):
            for i, l in enumerate(layers[1:]):
                deep_inputs = tf.contrib.layers.fully_connected(inputs=deep_inputs, num_outputs=l,
                                                                activation_fn=tf.nn.relu,
                                                                weights_regularizer=tf.contrib.layers.l2_regularizer(
                                                                    l2_reg),
                                                                scope='mlp%d' % i)
                # deep_inputs = BaseNMF.add_layer(deep_inputs,l,activation=tf.nn.relu,sub_name="mlp{}".format(i+1))

        # -----build concat layer -----------
        deep_inputs = tf.concat([mf_inputs, deep_inputs], 1)

        # ------build output layer-------------
        with tf.variable_scope("Out"):
            # deep_inputs = BaseNMF.add_layer(deep_inputs,1,activation=None,sub_name="out")
            deep_inputs = tf.contrib.layers.fully_connected(inputs=deep_inputs, num_outputs=1,
                                                            activation_fn=tf.identity,
                                                            weights_regularizer=tf.contrib.layers.l2_regularizer(
                                                                l2_reg))

        pred = deep_inputs

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
        rmse = tf.sqrt(tf.reduce_mean(tf.squared_difference(pred, labels)))
        loss = rmse + \
               l2_reg * tf.nn.l2_loss(W_mlp_user) + \
               l2_reg * tf.nn.l2_loss(W_mlp_item) + \
               l2_reg * tf.nn.l2_loss(W_mf_user) + \
               l2_reg * tf.nn.l2_loss(W_mf_item)

        # Provide an estimator spec for `ModeKeys.EVAL`
        eval_metric_ops = {
            "rmse": tf.metrics.root_mean_squared_error(labels, pred)
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
        self.model = tf.estimator.Estimator(model_fn=NeuMF.model_fn, model_dir=model_dir, params=mf_params,
                                            config=run_config)

    def train_and_evaluate(self, tr_files, va_files):
        train_spec = tf.estimator.TrainSpec(input_fn=lambda: NeuMF.input_fn(tr_files,
                                                                            num_epochs=mf_params["num_epochs"],
                                                                            batch_size=mf_params["batch_size"]))

        eval_spec = tf.estimator.EvalSpec(input_fn=lambda: NeuMF.input_fn(tr_files,
                                                                          num_epochs=1,
                                                                          batch_size=mf_params["batch_size"]))
        tf.estimator.train_and_evaluate(estimator=self.model,
                                        train_spec=train_spec,
                                        eval_spec=eval_spec)

    def train(self, tr_files, va_files):
        self.model.train(input_fn=lambda: NeuMF.input_fn(tr_files,
                                                         num_epochs=mf_params["num_epochs"],
                                                         batch_size=mf_params["batch_size"]))

    def evaluate(self, va_files):
        self.model.evaluate(input_fn=lambda: NeuMF.input_fn(va_files, num_epochs=1, batch_size=mf_params["batch_size"]))

    def predict(self, te_files, isSave=False, numToSave=10):
        P_G = self.model.predict(input_fn=lambda: NeuMF.input_fn(te_files, num_epochs=1, batch_size=1),
                                 predict_keys="prob")
        if isSave:
            with open(te_files, 'r') as f1, open('sample.unitest', "w") as f2:
                for i in range(numToSave):
                    sample = f1.readline()
                    result = next(P_G)
                    pred = str(result['prob'])
                    f2.write('\t'.join([pred, sample]))
            return None
        else:
            return P_G