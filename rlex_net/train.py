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
flag.DEFINE_list("ner_procedure",["pcnn","mi_att"], "layer should be include, order is important"
                                                    "ner处理过程使用的算法，按顺序搭建网络"
                                                    "正确的包括 pcnn,mi_att;"
                                                    "birnn,mi_att"
                                                    "transformer,mi_att2")
# tf.flags.DEFINE_string("label_voc_file_path", None, "tfrecord中的标签词的字典文件地址，为了兼容spark，目录下唯一text为file")
tf.flags.DEFINE_string("label_name", "label", "label's name in tfrecord input files,tfrecord中的标签的名字")
tf.flags.DEFINE_float("learning_rate", 0.01, '学习率.')
tf.flags.DEFINE_string("pos_name", "pos", "name of word's postition info in tfrecord input files when using pcnn, tfrecord中的position向量列的名字")
tf.flags.DEFINE_integer("embed_size",100,"如果使用预先训练的embedding，此参数无效，即embedding_file_path 不为None")
tf.flags.DEFINE_string("embedding_file_path",None,"optional, pre-trainning wordembedding path ,可选，预训练的embedding文件路径，包括embedding和vocabulary 2个文件，如果不为none，embed_size，feature_voc_file_path参数不起作用")
tf.flags.DEFINE_string("feature_name", "features", "feature's name in tfrecord input files,tfrecord中的特征的名字")
# tf.flags.DEFINE_string("feature_voc_file_path", None, "tfrecord中的特征词的字典文件地址，为了兼容spark，目录下唯一text为file")
tf.flags.DEFINE_integer("filters",64,"cnn kerenl numbers, cnn卷积核的个数")
tf.flags.DEFINE_string("cnn_kernel_size","[2,3,5]","cnn kernel length, cnn 卷积核的长度列表，类似于ngram")
tf.flags.DEFINE_string('log_level', 'INFO', 'tensorflow训练时的日志打印级别， 取值分别为，DEBUG，INFO,WARN,ERROR')
tf.flags.DEFINE_string('data_dir', 'd:\\cnn\\tfrecord\\', 'path of trainning,eval data,训练数据存放路径，支持hdfs')
tf.flags.DEFINE_string('model_dir', 'd:\\cnn\\model\\', 'path for saving model and checkpoint files, 保存dnn模型文件的路径，支持hdfs')
tf.flags.DEFINE_integer('batch_size', 64, '一批数量样本的数量')
tf.flags.DEFINE_list("gpu_cores",None,"multi-gpu config, gpu ids list,例如[0,1,2,3]，在当个GPU机器的情况，使用的哪些核来训练")
tf.flags.DEFINE_integer("check_steps", 300,'after how many steps, stopped saving check point and eval, 保存训练中间结果的间隔，也是evalutation的间隔')
tf.flags.DEFINE_integer('max_steps', 1000, 'after how many steps, stopped and saving model,训练模型最大的批训练次数，在model_dir不变的情况下重复训练'
                                               '，达到max_step后，不再继续训练，或者增加max_step，或者更换model_dir, 再继续训练')

tf.flags.DEFINE_integer("rnn_hidden_size",100,"birnn's hidden size when using birnn model, 当使用lstm 时候有效， lstm内部隐藏层大小")
tf.flags.DEFINE_string("distance_name","distance_name","when using birnn, distance from entity A and B are counted, the name of this feature in tfrecord,只在rnn的时候存在，feature的一部分，和语料tensor的结构相同，每个词的内容被一个2个元素的位置代替")
tf.flags.DEFINE_float("drop_out_rate", 0.5, "dropout概率，范围是0至1。例如rate=0.1会将输入Tensor的内容dropout10%。")


tf.flags.DEFINE_integer("enable_ema",0,"if exponent moving average is used, 是否启动指数移动平均来计算参数")

tf.flags.DEFINE_integer("num_classes", 0, "how many relations are there.有多少个不同的类别")
tf.flags.DEFINE_integer("num_features", 0, "how word are there, 有多少个不同的特征值")
tf.flags.DEFINE_integer("is_multi_label", 0, "is a mutli_relation ? 是否multi_label。值为0时，表示一条数据有且仅有一个关系label。而值为1时，表示一条数据可能对应多种关系label。")

FLAGS = flag.FLAGS
"""
 程序的入口，这个程序功能是从多句话中，分析两个（命名实体）关系的，
 比如
 “在今年4月， A公司以2亿美元的价格完成了对b公司100%的股权收购”
 “在收购以后，A公司对B公司的管理层进行了大范围的更换”
 从以上2句话来判断A公司和B公司的关系  关系有一定的范围“收购？控股？子公司？。。。。”
 
 pcnn的这个算法选用 ["pcnn","mi_att"] 这个流程,每个子项对应着一个网络层
 
"""
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

    # 加载数据，SparkInput 包含了数据的加载流程，即训练的数据从哪里来的，包括哪些项
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

    """
    定义estimator
    estimator，是整个训练的入口类,负责启动、管理整个模型训练过程
    同时也负责在训练过程中评估模型的质量
    最后负责把模型保存起来
    model_fn： 整个神经网络模型是如何构建的都在其中
    model_dir： 模型保存的地址，训练过程中的中间结果也在其中
    config：训练过程中，系统级的参数配置，比如是否使用gpu，训练多少轮保存一次中间结果等等
    params：应用级别的参数集合
    """
    estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=FLAGS.model_dir, config=config, params=params)

    #训练数据的路径
    train_data_dir = input.get_data_dir(tf.estimator.ModeKeys.TRAIN, params)
    #评估数据的路径
    eval_data_dir = input.get_data_dir(tf.estimator.ModeKeys.EVAL, params)

    # hook，即钩子， 用于每次评估的时候，执行，功能是把模型参数的训练值由原始值替换成移动平均值，这是一种优化手段
    hook = [] if not params.enable_ema else [LoadEMAHook(params.model_dir,0.99)]

    #每次保存中间训练结果的时候，执行EvalListener，用于评估训练的结果
    #评估数据还是input_fn 中来， 2个评估，使用训练数据评估和使用评估数据评估
    #hook的最用见上面
    listeners = [
        EvalListener(estimator, lambda: input.input_fn(mode = tf.estimator.ModeKeys.EVAL, params=params, data_dir=train_data_dir), name="train_data",hook = hook),
        EvalListener(estimator, lambda: input.input_fn(mode = tf.estimator.ModeKeys.EVAL, params=params, data_dir=eval_data_dir),hook = hook)
    ]

    #定义训练数据读入流，执行train_input_fn，可以得到一个特殊的tensor，在每次训练的step中，tensor里的内容都会变为数据流中的下一段，具体在input_fn中编码
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

        """
        开始执行训练，
        train_input_fn 是数据来源
        max_steps： 是最多训练多少步就停下来
        saving_listeners：是保存中间结果之后，自定义做哪些事情， 这里的listeners是会对中间结果做评估，看看模型的效果如何
       """
        estimator.train(train_input_fn, max_steps=FLAGS.max_steps, saving_listeners=listeners)
        dir = estimator.export_savedmodel(tf.flags.FLAGS.model_dir, input.get_input_reciever_fn())
        tf.logging.warn("save model to %s" % dir)

    for listener in listeners:
        print(listener.name)
        print(listener.history)



if __name__ == "__main__":
    tf.app.run(main, argv=None)


