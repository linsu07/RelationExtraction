import tensorflow as tf
import os
from tensorflow.python.estimator.export.export import build_raw_serving_input_receiver_fn
from rlex_net.parameters import user_params
from tensorflow.python.ops.lookup_ops import index_table_from_file
'''
定义如何从数据文件中得到训练用的输入tensor流
数据文件可以认为表格形式，
row 是一行一行的数据
column 是每个数据sample分为哪些项
'''

class SparkInput():
    def __init__(self, params: user_params):
        self.feature_name = params.feature_name
        self.pos_name = params.pos_name
        '''
        具体定义每个数据项的数据类型，tensorflow根据这些定义来解析文件
        label_name 是一个数字，代表一个具体归属类别， 神经网络的训练就是要把样本数据训练预测到这个类别
        weightColName： 这个没用到，是每个数据样本在训练过程中的重要程度，如果有的类别数据过少，可以调整它来增加权重
        '''
        self.conttext_spec = {
            params.label_name:tf.FixedLenFeature(shape=[1], dtype=tf.int64, default_value=0),
            params.weightColName: tf.FixedLenFeature(shape=[1], dtype=tf.float32, default_value=1.0)
        }
        """
        如果一行数据属于多个类别，则label_name 项就一个数组，包括多个数字。
        """
        if params.is_multi_label:
            self.conttext_spec[params.label_name] = tf.VarLenFeature(dtype=tf.int64)
        """
        feature_name 代表输入的文本，具体格式是多维数组[句子的个数][每个句子词的个数]
        pos_name： 代表输入文本中每个词的位置信息，在pcnn中，每个词的位置是 0，1，2，3 中一个
        """
        self.sequence_spec = {
            params.feature_name:tf.VarLenFeature(tf.int64),
            params.pos_name:tf.VarLenFeature(tf.int64)
        }

    def get_data_dir(self, mode: tf.estimator.ModeKeys, params: user_params):
        return os.path.join(params.data_dir, "train") if mode == tf.estimator.ModeKeys.TRAIN else os.path.join(params.data_dir, "evaluation")


    def input_fn(self, mode: tf.estimator.ModeKeys, params: user_params, data_dir):
        '''
            在estimator 的train方法中调用此方法获得tensor数据流
        :param mode:  在训练或者是评估或者是预测的模式下
        :param params: 预先设置的超参数
        :param data_dir: 数据来源地址
        :return: 训练的tensor数据
        '''
        file_paths = tf.gfile.Glob(os.path.join(data_dir, "part-r-*"))
        '''把训练数据文件集合装载到Dataset高级类中，方便输出，Dataset负责多线程打开文件，读入数据流等操作'''
        data_set = tf.data.TFRecordDataset(file_paths, buffer_size=10 * 1024 * 1024)

        '''对一行数据如何解析的操作'''
        def parse(raw):
            #根据context_features，sequence_features的定义，把数据解析出来放入context_dic,seq_dic 2个 dictionary的对象中
            #注意每个对象都是tensorflow的 sparse tensor类型，以下操作把 sparse tensor 转成 dense tensor
            context_dic,seq_dic = tf.parse_single_sequence_example(serialized=raw, context_features=self.conttext_spec, sequence_features=self.sequence_spec)
            if params.is_multi_label:
                # label
                multi_label = tf.cast(tf.sparse_tensor_to_dense(context_dic.get(params.label_name), default_value=0), tf.int32)
                # multi_hot label, NA's id is -1, so after one hot, It's label is all zeros
                label_string = tf.cast(tf.reduce_sum(tf.one_hot(multi_label, params.nClasses, axis=-1, name="label_onehot"), axis=-2), tf.int32)
            else:
                label_string = tf.cast(context_dic.get(params.label_name), tf.int32)

            seq_dic[params.weightColName] = context_dic.get(params.weightColName)
            seq_dic[params.feature_name] = tf.cast(tf.sparse_tensor_to_dense(seq_dic[params.feature_name], default_value=1), tf.int32)
            seq_dic[params.pos_name] = tf.cast(tf.sparse_tensor_to_dense(seq_dic[params.pos_name], default_value=0), tf.int32)

            return seq_dic, label_string

        # 因为每次训练都是一批数据，批中的数据句子个数，句子的长度都不一样，所以要加入padding，保证数据统一
        padd_batch_model = ({params.feature_name: [None, None],
                         params.pos_name:[None, None],
                         params.weightColName: [None]},
                        [None])
        padd_value = ({params.feature_name: 0,
                   params.pos_name:0,
                   params.weightColName: 1.0},
                  0)
        if params.is_multi_label:
            padd_batch_model[1][0] = params.nClasses

        if mode == tf.estimator.ModeKeys.TRAIN:
            '''
            repeat: 数据文件循环读的次数，None表示无限循环
            shuffle： 数据打散的空间是多大
            map：原始数据如何映射到有格式的tensor
            padded_batch： 把单条数据按batch_size集合成一批数据，padd_value，数据长度不一的时候，如何填充空白
            '''
            data_set = data_set.repeat(None).shuffle(buffer_size=20 * 1000) \
                .map(parse).padded_batch(params.batch_size, padd_batch_model, padd_value)#.batch(params.batch_size)#.prefetch(buffer_size=None)

        elif mode == tf.estimator.ModeKeys.EVAL:
            data_set = data_set.repeat(1) \
                .map(parse).padded_batch(2,padd_batch_model, padd_value)#.batch(params.batch_size)#.prefetch(buffer_size=None)
        '''返回的data_set就是tensor的数据流了'''
        return data_set#.make_initializable_iterator()

    def get_input_reciever_fn(self):

        feature = tf.placeholder(dtype=tf.int32, shape=[None, None, None], name="feature_tensor")
        pos_name = tf.placeholder(dtype=tf.int32, shape=[None, None, None], name = "pos_tensor")
        receiver_tensors = {self.feature_name: feature, self.pos_name: pos_name}

        return build_raw_serving_input_receiver_fn(receiver_tensors)

