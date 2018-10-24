import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

from rlex_net.parameters import user_params

'''
  为了验证AttentionProject2的正确性，
  特将birnn的输出，由
  [batch_size, sequence_num, features_num]
  改为
  [batch_size, sequence_num, sequence_len, features_num]
'''
class BIRnnLayer2(tf.layers.Layer):
    def __init__(self, params:user_params,feature_size:int, is_trainning=True, name="birnn2", dtype=tf.float32):
        super(BIRnnLayer2, self).__init__(is_trainning, name, dtype)
        self.params = params
        self.feature_size = feature_size

    def build(self, _):
        self.fw_cell = tf.nn.rnn_cell.LSTMCell(
            self.params.rnn_hidden_size
            ,initializer=xavier_initializer()
            ,name = "fw_cell"
        )
        if self.trainable:
            self.fw_cell = tf.nn.rnn_cell.DropoutWrapper(self.fw_cell,1.0-self.params.drop_out_rate)
        self.bw_cell = tf.nn.rnn_cell.LSTMCell(
            self.params.rnn_hidden_size
            ,initializer=xavier_initializer()
            ,name = "bw_cell"
        )
        if self.trainable:
            self.bw_cell = tf.nn.rnn_cell.DropoutWrapper(self.bw_cell,1.0-self.params.drop_out_rate)
        self.built = True

    def call(self, inputs, **kwargs):
        feature = inputs[self.params.feature_name]
        sentence_number = tf.shape(feature)[1]
        time_steps = tf.shape(feature)[2]
        feature_size =self.feature_size
        length = tf.cast(inputs["sen_length"], dtype=tf.int32, name="sentence_length")

        feature = tf.reshape(feature,[-1,time_steps,feature_size])
        length = tf.reshape(length,[-1])
        (output_fw, output_bw),states = tf.nn.bidirectional_dynamic_rnn(self.fw_cell
                                                   ,self.bw_cell
                                                   ,feature
                                                   ,sequence_length=length
                                                   ,dtype=tf.float32)
        # output_fw = [batch_size, max_time, cell_fw.output_size]
        # output_bw = [batch_size, max_time, cell_bw.output_size]
        bi_outputs = tf.concat([output_fw, output_bw], axis=-1)
        bi_outputs = tf.reshape(bi_outputs,[-1,sentence_number, time_steps, 2*self.params.rnn_hidden_size])
        inputs[self.params.feature_name] = bi_outputs
        return inputs
