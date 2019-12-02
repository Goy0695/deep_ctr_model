import numpy as np
import tensorflow as tf


class MF:

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

    def model_fn(features, labels, mode, params):
        """Bulid Model function f(x) for Estimator."""
        user_size = params["user_cnt"]
        item_size = params["item_cnt"]
        factor_size = params["factor"]
        l2_reg = params["l2_reg"]
        learning_rate = params["learning_rate"]
        mu = params['mu']

        # ------bulid weights------
        W_user = tf.get_variable(name='user_embedding', shape=[user_size, factor_size],
                                 initializer=tf.glorot_normal_initializer())
        W_item = tf.get_variable(name='item_embedding', shape=[item_size, factor_size],
                                 initializer=tf.glorot_normal_initializer())
        W_user_bias = tf.concat([W_user, tf.ones([user_size, 1], dtype=tf.float32, name='user_bias')], 1)
        W_item_bias = tf.concat([tf.ones([item_size, 1], dtype=tf.float32, name='item_bias'), W_item], 1)

        # ------build (user,item) pair-------
        user_ids = features['user_id']
        item_ids = features['item_id']

        # ------build f(x)------
        user_embedding = tf.nn.embedding_lookup(W_user_bias, user_ids)  # None*(k+1)
        item_embedding = tf.nn.embedding_lookup(W_item_bias, item_ids)  # None*(k+1)
        pred = tf.add(tf.reduce_sum(tf.multiply(user_embedding, item_embedding), 1), mu)  # None * 1

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
               l2_reg * tf.nn.l2_loss(W_user) + \
               l2_reg * tf.nn.l2_loss(W_item)

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
        self.model = tf.estimator.Estimator(model_fn=MF.model_fn, model_dir=model_dir, params=mf_params,
                                            config=run_config)

    def train(self, tr_files, va_files):
        self.model.train(input_fn=lambda: MF.input_fn(tr_files, num_epochs=mf_params["num_epochs"],
                                                      batch_size=mf_params["batch_size"]))

    def train_and_evaluate(self, tr_files, va_files):
        train_spec = tf.estimator.TrainSpec(input_fn=lambda: MF.input_fn(tr_files,
                                                                         num_epochs=mf_params["num_epochs"],
                                                                         batch_size=mf_params["batch_size"]))

        eval_spec = tf.estimator.EvalSpec(input_fn=lambda: MF.input_fn(tr_files,
                                                                       num_epochs=1,
                                                                       batch_size=mf_params["batch_size"]))
        tf.estimator.train_and_evaluate(estimator=self.model,
                                        train_spec=train_spec,
                                        eval_spec=eval_spec)

    def evaluate(self, va_files):
        self.model.evaluate(input_fn=lambda: MF.input_fn(va_files, num_epochs=1, batch_size=mf_params["batch_size"]))

    def predict(self, te_files, isSave=False, numToSave=10):
        P_G = self.model.predict(input_fn=lambda: MF.input_fn(te_files, num_epochs=1, batch_size=1),
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