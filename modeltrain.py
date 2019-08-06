import glob
import os
import random
import shutil
from datetime import date,timedelta
from Config import *
from src.base.model import basemodel
from src.base.model import FLAGS
from src.model.DeepFM import DeepFM
from src.model.FM import FM
from src.model.WideDeep import WideDeep
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def modeltrain():

    if FLAGS.modeltype in ["DeepFM","FM","FNN"]:
        train_format = "%s/tr*libsvm"
        val_format = "%s/va*libsvm"
        test_format = "%s/te*libsvm"
    else:
        train_format = "%s/tr*csv"
        val_format = "%s/va*csv"
        test_format = "%s/te*csv"

    tr_files = glob.glob(train_format % FLAGS.data_dir)
    random.shuffle(tr_files)
    print("tr_files:", tr_files)
    va_files = glob.glob(val_format % FLAGS.data_dir)
    print("va_files:", va_files)
    te_files = glob.glob(test_format % FLAGS.data_dir)
    print("te_files:", te_files)

    if FLAGS.clear_existing_model:
        try:
            shutil.rmtree(FLAGS.model_dir)
        except Exception as e:
            print(e, "at clear_existing_model")
        else:
            print("existing model cleaned at %s" % FLAGS.model_dir)

    model_params = {
        "field_size": FLAGS.field_size,
        "feature_size": FLAGS.feature_size,
        "embedding_size": FLAGS.embedding_size,
        "learning_rate": FLAGS.learning_rate,
        "batch_norm_decay": FLAGS.batch_norm_decay,
        "optimizer": FLAGS.optimizer,
        "l2_reg": FLAGS.l2_reg,
        "deep_layers": FLAGS.deep_layers,
        "dropout": FLAGS.dropout
    }

    config = tf.estimator.RunConfig().replace(session_config = tf.ConfigProto(device_count={'GPU':0, 'CPU':FLAGS.num_threads}),
                                              log_step_count_steps=FLAGS.log_steps, save_summary_steps=FLAGS.log_steps)

    print("setitng {} model....".format(FLAGS.modeltype))

    if FLAGS.modeltype == "DeepFM":
        dmodel = DeepFM(model_params,config)
    elif FLAGS.modeltype == "WideDeep":
        dmodel = WideDeep(model_params,config)
        dmodel.set_feature_columns(FLAGS.data_dir)
    elif FLAGS.modeltype == "FM":
        dmodel = FM(model_params,config)
    else:
        print("Wrong model type!")

    print("compiling {} model....".format(FLAGS.modeltype))
    dmodel.compile()

    print("training {} model....".format(FLAGS.modeltype))
    dmodel.train(tr_files,FLAGS)


    print("evaluating {} model....".format(FLAGS.modeltype))
    dmodel.evaluate(va_files,FLAGS)

    #dmodel.predict(te_files,FLAGS)

    #dmodel.save()

if __name__ == "__main__":
    if FLAGS.dt_dir == "":
        FLAGS.dt_dir = (date.today() + timedelta(-1)).strftime('%Y%m%d')
    FLAGS.model_dir = FLAGS.model_dir + FLAGS.dt_dir
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(modeltrain())




