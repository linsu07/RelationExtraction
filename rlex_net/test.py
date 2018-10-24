import sys
sys.path.insert(0,"/disk1/liutianyu")
from rlex_net.parameters import user_params, enrich_hyper_parameters
from rlex_net.input import SparkInput
from rlex_net.train import main
import tensorflow as tf
FLAGS = tf.flags.FLAGS
if __name__ == "__main__":

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

    # debug
    tf.enable_eager_execution()
    from tensorflow.contrib.eager.python.tfe import Iterator
    i = 0
    for item in Iterator(input.input_fn(mode = tf.estimator.ModeKeys.TRAIN, params=params, data_dir=input.get_data_dir(tf.estimator.ModeKeys.TRAIN, params))):
        words = item[0][params.feature_name]
        pos = item[0][params.pos_name]
        label = item[1]
        print("########################")
        print(words)
        print(pos)
        print(label)
        if i == 20:
            break
        i += 1
