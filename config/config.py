import tensorflow as tf
root_path = "/home/yu2.guo/notebooks/deep_ctr_model_v1"
app = "HK_train"
model_type = 'WideDeep'
data_path = root_path+"/data/" + app
model_path = root_path+"/model/"
FLAGS = tf.app.flags.FLAGS
#兼容jupyter
tf.app.flags.DEFINE_string('f', '', 'kernel')
#输入数据
tf.app.flags.DEFINE_integer("feature_size", 18165, "Number of features")
tf.app.flags.DEFINE_integer("field_size", 39, "Number of fields")
tf.app.flags.DEFINE_integer("continuous_field_size", 13, "Number of continuous feature fields")
tf.app.flags.DEFINE_integer("category_field_size", 26, "Number of category feature fields")
#模型类型
tf.app.flags.DEFINE_string("modeltype", model_type, "model type {DeepFM,WideDeep...}")
#打印日志间隔
tf.app.flags.DEFINE_integer("log_steps", 500, "save summary every steps")
#输入输出路径
tf.app.flags.DEFINE_string("data_dir", data_path, "data dir")
tf.app.flags.DEFINE_string("model_dir", model_path+'/tmp', "model check point dir")
tf.app.flags.DEFINE_string("servable_model_dir", model_path+'/serving/'+app+'.'+model_type+'.model', "export servable model")
tf.app.flags.DEFINE_boolean("clear_existing_model", True, "clear existing model or not")


