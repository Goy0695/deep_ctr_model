import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

#兼容jupyter
tf.app.flags.DEFINE_string('f', '', 'kernel')

#分布式训练参数
tf.app.flags.DEFINE_integer("num_threads", 8, "Number of threads")

#特征参数
tf.app.flags.DEFINE_integer("feature_size", 117581, "Number of features")
tf.app.flags.DEFINE_integer("field_size", 39, "Number of fields")
tf.app.flags.DEFINE_integer("continuous_field_size", 13, "Number of continuous feature fields")
tf.app.flags.DEFINE_integer("category_field_size", 26, "Number of category feature fields")

#模型结构、损失函数、优化算法相关参数
tf.app.flags.DEFINE_string("modeltype", 'FM', "model type {DeepFM,WideDeep...}")
tf.app.flags.DEFINE_string("loss_type", 'log_loss', "loss type {square_loss, log_loss}")
tf.app.flags.DEFINE_string("optimizer", 'Adam', "optimizer type {Adam, Adagrad, GD, Momentum}")
tf.app.flags.DEFINE_string("deep_layers", '16,8', "deep layers")
tf.app.flags.DEFINE_string("dropout", '0.5,0.5', "dropout rate")
tf.app.flags.DEFINE_boolean("batch_norm", False, "perform batch normaization (True or False)")
tf.app.flags.DEFINE_float("batch_norm_decay", 0.9, "decay for the moving average(recommend trying decay=0.9)")

#模型训练相关参数
tf.app.flags.DEFINE_integer("embedding_size", 8, "Embedding size")
tf.app.flags.DEFINE_integer("num_epochs", 2, "Number of epochs")
tf.app.flags.DEFINE_integer("batch_size", 64, "Number of batch size")
tf.app.flags.DEFINE_integer("log_steps", 1000, "save summary every steps")
tf.app.flags.DEFINE_float("learning_rate", 0.01, "learning rate")
tf.app.flags.DEFINE_float("l2_reg", 0.0001, "L2 regularization")

#输入输出路径
tf.app.flags.DEFINE_string("data_dir", 'D:\MyConfiguration\yu2.guo.TCENT\PycharmProjects\Deep_net\data', "data dir")
tf.app.flags.DEFINE_string("dt_dir", '', "data dt partition")
tf.app.flags.DEFINE_string("model_dir", 'D:\MyConfiguration\yu2.guo.TCENT\PycharmProjects\Deep_net\model', "model check point dir")
tf.app.flags.DEFINE_string("servable_model_dir", 'D:\MyConfiguration\yu2.guo.TCENT\PycharmProjects\DeepModel\model', "export servable model for TensorFlow Serving")
tf.app.flags.DEFINE_string("task_type", 'train', "task type {train, infer, eval, export}")
tf.app.flags.DEFINE_boolean("clear_existing_model", False, "clear existing model or not")


