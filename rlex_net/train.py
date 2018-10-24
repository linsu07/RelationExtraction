import tensorflow as tf
from tensorflow.python.estimator.run_config import RunConfig, TaskType
import sys
sys.path.insert(0,"/disk1/liutianyu")
from common import MyTraining
from common.listeners import EvalListener, LoadEMAHook
from rlex_net.networks.model import model_fn
from rlex_net.parameters import user_params, enrich_hyper_parameters
from rlex_net.input import SparkInput

flag = tf.flags
flag.DEFINE_list("ner_procedure",["pcnn","mi_att"], ""
                                                    "ner处理过程使用的算法，按顺序搭建网络"
                                                    "正确的包括 pcnn,mi_att;"
                                                    "birnn,mi_att"
                                                    "transformer,mi_att2")
# tf.flags.DEFINE_string("label_voc_file_path", None, "tfrecord中的标签词的字典文件地址，为了兼容spark，目录下唯一text为file")
tf.flags.DEFINE_string("label_name", "label", "tfrecord中的标签的名字")
tf.flags.DEFINE_float("learning_rate", 0.01, '学习率.')
tf.flags.DEFINE_string("pos_name", "pos", "tfrecord中的position向量列的名字")
tf.flags.DEFINE_integer("embed_size",100,"如果使用预先训练的embedding，此参数无效，即embedding_file_path 不为None")
tf.flags.DEFINE_string("embedding_file_path",None,"可选，预训练的embedding文件路径，包括embedding和vocabulary 2个文件，如果不为none，embed_size，feature_voc_file_path参数不起作用")
tf.flags.DEFINE_string("feature_name", "features", "tfrecord中的特征的名字")
# tf.flags.DEFINE_string("feature_voc_file_path", None, "tfrecord中的特征词的字典文件地址，为了兼容spark，目录下唯一text为file")
tf.flags.DEFINE_integer("filters",64,"cnn卷积核的个数")
tf.flags.DEFINE_string("cnn_kernel_size","[2,3,5]","cnn 卷积核的长度列表，类似于ngram")
tf.flags.DEFINE_string('log_level', 'INFO', 'tensorflow训练时的日志打印级别， 取值分别为，DEBUG，INFO,WARN,ERROR')
tf.flags.DEFINE_string('data_dir', 'd:\\cnn\\tfrecord\\', '训练数据存放路径，支持hdfs')
tf.flags.DEFINE_string('model_dir', 'd:\\cnn\\model\\', '保存dnn模型文件的路径，支持hdfs')
tf.flags.DEFINE_integer('batch_size', 64, '一批数量样本的数量')
tf.flags.DEFINE_list("gpu_cores",None,"例如[0,1,2,3]，在当个GPU机器的情况，使用的哪些核来训练")
tf.flags.DEFINE_integer("check_steps", 300,'保存训练中间结果的间隔，也是evalutation的间隔')
tf.flags.DEFINE_integer('max_steps', 1000, '训练模型最大的批训练次数，在model_dir不变的情况下重复训练'
                                               '，达到max_step后，不再继续训练，或者增加max_step，或者更换model_dir, 再继续训练')

tf.flags.DEFINE_integer("rnn_hidden_size",100,"当使用lstm 时候有效， lstm内部隐藏层大小")
tf.flags.DEFINE_string("distance_name","distance_name","只在rnn的时候存在，feature的一部分，和语料tensor的结构相同，每个词的内容被一个2个元素的位置代替")
tf.flags.DEFINE_float("drop_out_rate", 0.5, "dropout概率，范围是0至1。例如rate=0.1会将输入Tensor的内容dropout10%。")


tf.flags.DEFINE_integer("enable_ema",0,"是否启动指数移动平均来计算参数")

tf.flags.DEFINE_integer("transfromer_layers",2,"使用google 的transformer做encoder的层数")
tf.flags.DEFINE_integer("transfromer_head_number",8,"使用google 的transformer每一层的head的数量")
tf.flags.DEFINE_integer("tansformer_d_model",320,"使用google 的transformer每一层feature的维度，最好和embedding_size相等")
tf.flags.DEFINE_integer("tansformer_shareweight",0,"使用google 的transformer每一层是否共享参数")
tf.flags.DEFINE_integer("num_classes", 0, "有多少个不同的类别")
tf.flags.DEFINE_integer("num_features", 0, "有多少个不同的特征值")
tf.flags.DEFINE_integer("is_multi_label", 0, "是否multi_label。值为0时，表示一条数据有且仅有一个关系label。而值为1时，表示一条数据可能对应多种关系label。")

FLAGS = flag.FLAGS

def main(_):
    print(FLAGS.ner_procedure)
    params = user_params(ner_procedure=FLAGS.ner_procedure, label_voc_file_path=None,
                         label_name=FLAGS.label_name,
                         learning_rate=FLAGS.learning_rate, pos_name=FLAGS.pos_name,
                         embed_size=FLAGS.embed_size,
                         embedding_file_path=FLAGS.embedding_file_path,
                         feature_name = FLAGS.feature_name, feature_voc_file_path=None,
                         cnn_kernel_size=eval(FLAGS.cnn_kernel_size), filters=FLAGS.filters,
                         rnn_hidden_size=FLAGS.rnn_hidden_size,
                         distance_name=FLAGS.distance_name,
                         data_dir=FLAGS.data_dir,
                         model_dir=FLAGS.model_dir, batch_size=FLAGS.batch_size, drop_out_rate=FLAGS.drop_out_rate
                         ,enable_ema=FLAGS.enable_ema
                         ,transfromer_layers = FLAGS.transfromer_layers
                         ,transfromer_head_number=FLAGS.transfromer_head_number
                         ,tansformer_d_model = FLAGS.tansformer_d_model
                         ,tansformer_shareweight = FLAGS.tansformer_shareweight
                         ,gpu_cors = FLAGS.gpu_cores,num_classes=FLAGS.num_classes, num_features=FLAGS.num_features,
                         is_multi_label=FLAGS.is_multi_label)


    enrich_hyper_parameters(params)

    # 配置日志等级
    level_str = 'tf.logging.{}'.format(str(tf.flags.FLAGS.log_level).upper())
    tf.logging.set_verbosity(eval(level_str))

    # 加载数据
    input = SparkInput(params)


    # estimator运行环境配置
    session_config = tf.ConfigProto()
    session_config.allow_soft_placement = True
    session_config.gpu_options.per_process_gpu_memory_fraction = 0.9
    session_config.gpu_options.allow_growth = True
    # session_config.log_device_placement = True

    if FLAGS.gpu_cores:
        gpu_cors = tuple(FLAGS.gpu_cores)
        devices =  ["/device:GPU:%d" % int(d) for d in gpu_cors]
        tf.logging.warn("using device: " + " ".join(devices))
        distribution = tf.contrib.distribute.MirroredStrategy(devices = devices)

        tf.logging.warn("in train.py, distribution")
        tf.logging.warn(distribution._devices)

        config = RunConfig(save_checkpoints_steps=FLAGS.check_steps,train_distribute=distribution, keep_checkpoint_max=2, session_config=session_config)
    else:

        config = RunConfig(save_checkpoints_steps=FLAGS.check_steps, keep_checkpoint_max=2, session_config=session_config)


    estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=FLAGS.model_dir, config=config, params=params)

    train_data_dir = input.get_data_dir(tf.estimator.ModeKeys.TRAIN, params)
    eval_data_dir = input.get_data_dir(tf.estimator.ModeKeys.EVAL, params)

    hook = [] if not params.enable_ema else [LoadEMAHook(params.model_dir,0.99)]

    listeners = [
        EvalListener(estimator, lambda: input.input_fn(mode = tf.estimator.ModeKeys.EVAL, params=params, data_dir=train_data_dir), name="train_data",hook = hook),
        EvalListener(estimator, lambda: input.input_fn(mode = tf.estimator.ModeKeys.EVAL, params=params, data_dir=eval_data_dir),hook = hook)
    ]

    def train_input_fn():
        return input.input_fn(mode = tf.estimator.ModeKeys.TRAIN, params=params, data_dir=train_data_dir)

    # gpu cluster
    if config.cluster_spec:
        train_spec = MyTraining.TrainSpec(train_input_fn, FLAGS.max_steps)
        eval_spec = MyTraining.EvalSpec(lambda: input.input_fn(mode = tf.estimator.ModeKeys.EVAL, params=params, data_dir=train_data_dir), steps=FLAGS.check_steps)
        MyTraining.train_and_evaluate(estimator, train_spec, eval_spec, listeners)
        if config.task_type == TaskType.CHIEF:
            model_dir = estimator.export_savedmodel(FLAGS.model_dir, input.get_input_reciever_fn())
            tf.logging.warn("save model to %s" % model_dir)

    # cpu solo
    else:
        from tensorflow.python import debug as tf_debug
        # debug_hook = [tf_debug.LocalCLIDebugHook(ui_type="readline")]
        # estimator.train(train_input_fn, max_steps=FLAGS.max_steps, saving_listeners=listeners, hooks=debug_hook)
        estimator.train(train_input_fn, max_steps=FLAGS.max_steps, saving_listeners=listeners)
        dir = estimator.export_savedmodel(tf.flags.FLAGS.model_dir, input.get_input_reciever_fn())
        tf.logging.warn("save model to %s" % dir)

    for listener in listeners:
        print(listener.name)
        print(listener.history)



if __name__ == "__main__":
    tf.app.run(main, argv=None)


