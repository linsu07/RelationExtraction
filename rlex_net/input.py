import tensorflow as tf
import os
from tensorflow.python.estimator.export.export import build_raw_serving_input_receiver_fn
from rlex_net.parameters import user_params
from tensorflow.python.ops.lookup_ops import index_table_from_file

class SparkInput():
    def __init__(self, params: user_params):
        self.feature_name = params.feature_name
        self.pos_name = params.pos_name
        self.conttext_spec = {
            params.label_name:tf.FixedLenFeature(shape=[1], dtype=tf.int64, default_value=0),
            params.weightColName: tf.FixedLenFeature(shape=[1], dtype=tf.float32, default_value=1.0)

        }
        if params.is_multi_label:
            self.conttext_spec[params.label_name] = tf.VarLenFeature(dtype=tf.int64)
        self.sequence_spec = {
            params.feature_name:tf.VarLenFeature(tf.int64),
            params.pos_name:tf.VarLenFeature(tf.int64)
        }

    def get_data_dir(self, mode: tf.estimator.ModeKeys, params: user_params):
        return os.path.join(params.data_dir, "train") if mode == tf.estimator.ModeKeys.TRAIN else os.path.join(params.data_dir, "evaluation")



    def input_fn(self, mode: tf.estimator.ModeKeys, params: user_params, data_dir):

        file_paths = tf.gfile.Glob(os.path.join(data_dir, "part-r-*"))
        data_set = tf.data.TFRecordDataset(file_paths, buffer_size=10 * 1024 * 1024)

        def parse(raw):
            context_dic,seq_dic = tf.parse_single_sequence_example(serialized=raw, context_features=self.conttext_spec, sequence_features=self.sequence_spec)
            if params.is_multi_label:
                # label 这里的default_value，表示数据没有类别时，应取类别NA
                multi_label = tf.cast(tf.sparse_tensor_to_dense(context_dic.get(params.label_name), default_value=0), tf.int32)
                # multi_hot label
                label_string = tf.cast(tf.reduce_sum(tf.one_hot(multi_label, params.nClasses, axis=-1, name="label_onehot"), axis=-2), tf.int32)
            else:
                label_string = tf.cast(context_dic.get(params.label_name), tf.int32)

            seq_dic[params.weightColName] = context_dic.get(params.weightColName)
            seq_dic[params.feature_name] = tf.cast(tf.sparse_tensor_to_dense(seq_dic[params.feature_name], default_value=1), tf.int32)
            seq_dic[params.pos_name] = tf.cast(tf.sparse_tensor_to_dense(seq_dic[params.pos_name], default_value=0), tf.int32)

            return seq_dic, label_string

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
            data_set = data_set.repeat(None).shuffle(buffer_size=20 * 1000) \
                .map(parse).padded_batch(params.batch_size, padd_batch_model, padd_value)#.batch(params.batch_size)#.prefetch(buffer_size=None)

        elif mode == tf.estimator.ModeKeys.EVAL:
            data_set = data_set.repeat(1) \
                .map(parse).padded_batch(2,padd_batch_model, padd_value)#.batch(params.batch_size)#.prefetch(buffer_size=None)

        return data_set#.make_initializable_iterator()

    def get_input_reciever_fn(self):

        feature = tf.placeholder(dtype=tf.int32, shape=[None, None, None], name="feature_tensor")
        pos_name = tf.placeholder(dtype=tf.int32, shape=[None, None, None], name = "pos_tensor")
        receiver_tensors = {self.feature_name: feature, self.pos_name: pos_name}

        return build_raw_serving_input_receiver_fn(receiver_tensors)

