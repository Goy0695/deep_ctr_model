import tensorflow as tf
root_path = "/home/yu2.guo/notebooks/deep_ctr_model_v1"
app = "xuanji"
model_type = 'FM'
data_path = root_path+"/data/" + app
model_path = root_path+"/model/"
FLAGS = tf.app.flags.FLAGS
#兼容jupyter
tf.app.flags.DEFINE_string('f', '', 'kernel')
#模型类型
tf.app.flags.DEFINE_string("modeltype", model_type, "model type {DeepFM,WideDeep...}")
#打印日志间隔
tf.app.flags.DEFINE_integer("log_steps", 3000, "log_step_count_steps")
tf.app.flags.DEFINE_integer("save_steps", 20000, "save summary every many steps")
tf.app.flags.DEFINE_integer("check_steps", 6000, "save checkpoint every many steps")
#输入输出路径
tf.app.flags.DEFINE_string("data_dir", data_path, "data dir")
tf.app.flags.DEFINE_string("model_dir", model_path+'/checkpoint', "model check point dir")
tf.app.flags.DEFINE_string("servable_model_dir", model_path+'/serving/'+app+'.'+model_type+'.model', "export servable model")
tf.app.flags.DEFINE_boolean("clear_existing_model", True, "clear existing model or not")


