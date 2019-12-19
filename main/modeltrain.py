import glob
import os
import random
import shutil
import tensorflow as tf
from config.LR_Config import lr_params
from config.DeepFM_Config import deepfm_params
from config.FM_Config import fm_params
from config.FFM_Config import ffm_params
from config.NFM_Config import nfm_params
from config.AFM_Config import afm_params
from config.GBDT_Config import gbdt_params
from config.GBDT_LR_Config import stack_gbdt_params, stack_lr_params
from config.GBDT_FM_Config import stack_gbdt_params, stack_fm_params
from config.WideDeep_Config import widedeep_params
from config.config import FLAGS
from src.model.DeepFM import DeepFM
from src.model.FM import FM
from src.model.FFM import FFM
from src.model.NFM import NFM
from src.model.AFM import AFM
from src.model.LR import LR
from src.model.GBDT import GBDT
from src.model.WideDeep import WideDeep
from src.model.GBDT_LR import GBDT_LR
from src.model.GBDT_FM import GBDT_FM

# 设置输出日志等级
"""
const int INFO = 0;            // base_logging::INFO;
const int WARNING = 1;         // base_logging::WARNING;
const int ERROR = 2;           // base_logging::ERROR;
const int FATAL = 3;           // base_logging::FATAL;
log信息共有四个等级，按重要性递增为：
INFO（通知）<WARNING（警告）<ERROR（错误）<FATAL（致命的）
"""
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
# tf.logging.set_verbosity(tf.logging.ERROR)

# 设置GPU（多GPU）
"""
目前机器就两个gpu,编号0,1
"""
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 设置gpu内存大小
# gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
# gpu_config=tf.ConfigProto(gpu_options=gpu_options)
# gpu_config.gpu_options.allow_growth = True

# 设置gpu运行方式
gpu_config = tf.compat.v1.ConfigProto(
    log_device_placement=True,
    inter_op_parallelism_threads=0,
    intra_op_parallelism_threads=0,
    allow_soft_placement=True)
gpu_config.gpu_options.visible_device_list = "0,1"
gpu_config.gpu_options.allow_growth = True
gpu_config.gpu_options.allocator_type = 'BFC'


# 单机多卡分布式训练
# mirrored_strategy = tf.contrib.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])

def modeltrain(clear_existing_model=True, log_level="INFO"):
    if log_level == "INFO":
        tf.logging.set_verbosity(tf.compat.v1.logging.INFO)
    elif log_level == "WARNING":
        tf.logging.set_verbosity(tf.compat.v1.logging.WARNING)
    elif log_level == "ERROR":
        tf.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    else:
        tf.logging.set_verbosity(tf.compat.v1.logging.FATAL)

    if FLAGS.modeltype in ["DeepFM", "FM", "FNN", "LR", "NFM", "AFM", "FFM"]:
        train_format = "%s/tr.libsvm"
        val_format = "%s/va.libsvm"
        test_format = "%s/te.libsvm"
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

    if clear_existing_model:
        try:
            shutil.rmtree(FLAGS.model_dir)
        except Exception as e:
            print(e, "at clear_existing_model")
        else:
            print("existing model cleaned at %s" % FLAGS.model_dir)

    run_config = tf.estimator.RunConfig().replace(save_checkpoints_steps=FLAGS.check_steps,
                                                  tf_random_seed=2019,
                                                  session_config=gpu_config,
                                                  log_step_count_steps=FLAGS.log_steps,
                                                  save_summary_steps=FLAGS.save_steps,
                                                  keep_checkpoint_max=20)

    print("setitng {} model....".format(FLAGS.modeltype))
    if FLAGS.modeltype == "DeepFM":
        dmodel = DeepFM(run_config, deepfm_params)
    elif FLAGS.modeltype == "WideDeep":
        dmodel = WideDeep(run_config, widedeep_params, FLAGS.data_dir + '/feature')
        dmodel.set_feature_columns(FLAGS.data_dir)
    elif FLAGS.modeltype == "FM":
        dmodel = FM(run_config, fm_params)
    elif FLAGS.modeltype == "XGB":
        dmodel = XGB(xgb_params)
    elif FLAGS.modeltype == "GBDT":
        dmodel = GBDT(gbdt_params)
    elif FLAGS.modeltype == "GBDT_LR":
        dmodel = GBDT_LR(run_config, stack_gbdt_params, stack_lr_params)
    elif FLAGS.modeltype == "GBDT_FM":
        dmodel = GBDT_FM(run_config, stack_gbdt_params, stack_fm_params)
    elif FLAGS.modeltype == "LR":
        dmodel = LR(run_config, lr_params)
    elif FLAGS.modeltype == "FFM":
        dmodel = FFM(run_config, ffm_params)
    elif FLAGS.modeltype == "NFM":
        dmodel = NFM(run_config, nfm_params)
    elif FLAGS.modeltype == "AFM":
        dmodel = AFM(run_config, afm_params)
    else:
        print("Wrong model type!")

    print("compiling {} model....".format(FLAGS.modeltype))
    dmodel.compile('./model/checkpoint/{}'.format(FLAGS.modeltype))

    print("training {} model....".format(FLAGS.modeltype))
    if FLAGS.isValidMon:
        dmodel.train_and_evaluate(tr_files, va_files)
    else:
        dmodel.train(tr_files, va_files)

    print("evaluating {} model....".format(FLAGS.modeltype))
    dmodel.evaluate(va_files)

    # dmodel.predict(te_files)
    # print("saving {} model....".format(FLAGS.modeltype))
    # dmodel.save(FLAGS.servable_model_dir)
    return dmodel


if __name__ == "__main__":
    modeltrain()




