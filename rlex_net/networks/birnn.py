import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

from rlex_net.parameters import user_params

'''
  * Created by linsu on 2018/8/1.
  * mailto: lsishere2002@hotmail.com
'''
class BIRnnLayer(tf.layers.Layer):
    def __init__(self, params:user_params,feature_size:int, is_trainning=True, name="pcnn_layer", dtype=tf.float32):
        super(BIRnnLayer, self).__init__(is_trainning, name, dtype)
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
        # self.losses.extend(self.fw_cell.losses())
        # self.losses.extend(self.bw_cell.losses())
        self.built = True

    def call(self, inputs, **kwargs):
        feature = inputs[self.params.feature_name]
        sentence_number = tf.shape(feature)[1]
        time_steps = tf.shape(feature)[2]
        feature_size =self.feature_size
        length = tf.cast(inputs["sen_length"], dtype=tf.int32, name="sentence_length")

        feature = tf.reshape(feature,[-1,time_steps,feature_size])
        length = tf.reshape(length,[-1])
        _,states = tf.nn.bidirectional_dynamic_rnn(self.fw_cell
                                        ,self.bw_cell
                                        ,feature
                                        ,sequence_length=length
                                        ,dtype=tf.float32)
        bi_states = tf.concat([states[0][1], states[1][1]],-1)
        bi_states = tf.reshape(bi_states,[-1,sentence_number,2*self.params.rnn_hidden_size])
        inputs[self.params.feature_name] = bi_states
        return inputs