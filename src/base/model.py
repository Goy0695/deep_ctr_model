import tensorflow as tf
from Config import *
from Config import FLAGS

class basemodel:

    def __init__(self,params,config):
        self.params = params
        self.config = config
        self.model = None

    def input_fn(filenames,batch_size=32, num_epochs=1, perform_shuffle=False):
        print('Parsing', filenames)

        #decode csv file for W&D
        def decode_csv(line):
            continus_sample = [[0.0] for i in range(FLAGS.continuous_field_size)]
            category_sample = [["0.0"] for i in range(FLAGS.category_field_size)]
            sample = [[0.0]] + continus_sample + category_sample
            item = tf.decode_csv(line,sample)
            feature = item[1:]
            label = tf.expand_dims(item[0],-1)
            indx = [str(item) for item in range(1,FLAGS.field_size+1)]
            return dict(zip(indx,feature)),label

        #decode libsvm file for DFM
        def decode_libsvm(line):
            columns = tf.string_split([line], ' ')
            labels = tf.string_to_number(columns.values[0], out_type=tf.float32)
            splits = tf.string_split(columns.values[1:], ':')
            id_vals = tf.reshape(splits.values,splits.dense_shape)
            feat_ids, feat_vals = tf.split(id_vals,num_or_size_splits=2,axis=1)
            feat_ids = tf.string_to_number(feat_ids, out_type=tf.int32)
            feat_vals = tf.string_to_number(feat_vals, out_type=tf.float32)
            return {"feat_ids": feat_ids, "feat_vals": feat_vals}, labels

        if FLAGS.modeltype == "WideDeep":
            dataset = tf.data.TextLineDataset(filenames).map(decode_csv, num_parallel_calls=10).prefetch(500000)
        else:
            dataset = tf.data.TextLineDataset(filenames).map(decode_libsvm, num_parallel_calls=10).prefetch(500000)
        if perform_shuffle:
            dataset = dataset.shuffle(buffer_size=256)
        # epochs from blending together.
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.batch(batch_size) # Batch size to use
        iterator = dataset.make_one_shot_iterator()
        batch_features, batch_labels = iterator.get_next()
        return batch_features, batch_labels

    def model_fn(features, labels, mode,params):
        return None

    def compile(self):
        return None

    def train(self,tr_files,FLAGS):
        params = self.params
        self.model.train(input_fn=lambda: basemodel.input_fn(tr_files,num_epochs=FLAGS.num_epochs, batch_size=FLAGS.batch_size))

    def evaluate(self,va_files,FLAGS):
        params = self.params
        self.model.evaluate(input_fn=lambda: basemodel.input_fn(va_files, num_epochs=1, batch_size=FLAGS.batch_size))

    def predict(self,te_files,FLAGS):
        P_G = self.model.predict(input_fn=lambda: basemodel.input_fn(te_files, num_epochs=1, batch_size=FLAGS.batch_size), predict_keys="prob")
        return P_G

    def save(self,output_dir):
        model = self.model
        return 0