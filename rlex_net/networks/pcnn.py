import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

'''
  * Created by linsu on 2018/7/26.
  * mailto: lsishere2002@hotmail.com
'''


class PcnnLayer(tf.layers.Layer):
    def __init__(self, params, is_trainning=True, name="pcnn_layer", dtype=tf.float32):
        super(PcnnLayer, self).__init__(is_trainning, name, dtype)
        self.params = params

    def build(self, _):
        self.cnns = []
        for size in self.params.cnn_kernel_size:
            # cnn_w = [height, width, in_channels, out_channels]
            cnn = tf.layers.Conv2D(filters=self.params.filters
                , kernel_size=[1, size]
                , padding="same"
                , kernel_initializer=xavier_initializer()
                , trainable=self.trainable
                , activation=tf.nn.relu6
                , name="cnn_"+str(size)
                , kernel_regularizer=tf.nn.l2_loss)
            if not tf.executing_eagerly():
                self.add_loss(cnn.losses)
            self.cnns.append(cnn)
        self.dropout = tf.layers.Dropout(rate=self.params.drop_out_rate, trainable=self.trainable, name="dropout")
        self.built = True

    def call(self, inputs, **kwargs):
        print("in pcnn call.....")
        docs = inputs[self.params.feature_name]
        positons = inputs[self.params.pos_name]

        seq_number = tf.gather(tf.shape(docs),2)
        sentence_number = tf.gather(tf.shape(docs),1)
        # docs = [batch_size * sentence_number, 1: in_channel, seq_number, embedding_size]
        docs = tf.reshape(docs,[-1,1,seq_number,self.params.embedding_size])
        # positions = [batch_size * sentence_number, 1, seq_number, 1, 3: pos_encode_size]
        positons = tf.reshape(positons,[-1,1,seq_number,1,3])

        if tf.executing_eagerly():
            print(" a doc is \r\n {}\r\n {}".format(docs,tf.shape(docs)))
            print(" a doc_pos is \r\n {}\r\n {}".format(positons,tf.shape(positons)))

        # modify by lty
        # 把单卷积核大小，改为多种卷积核大小。
        # old code ####################
        # conv = self.cnn(docs)
        ###############################

        convs = []
        for cnn in self.cnns:
            # conv = [batch_size * sentence_number, 1, seq_number, filters: out_channel]
            conv = cnn(docs)
            convs.append(conv)
        # conv = [batch_size * sentence_number, 1, seq_number, filters * len(cnn_kernel_size)]
        conv = tf.concat(convs, axis=-1)

        if tf.executing_eagerly():
            print(" a conv result is \r\n {}\r\n {}".format(conv,tf.shape(conv)))
        # conv_reshape = [batch_size * sentence_number, 1, seq_number, filters * len(cnn_kernel_size), 1]
        conv_reshape = tf.expand_dims(conv, axis=-1)
        # mul = [batch_size * sentence_number, 1, seq_number, filters * len(cnn_kernel_size), 3: pos_encode_size]
        mul = tf.multiply(conv_reshape,positons)
        if tf.executing_eagerly():
            print("tf.multiply(conv_reshape,pos_reshape) \r\n {}\r\n {}".format(mul,tf.shape(mul)))
        # conv_pooling = [batch_size * sentence_number, 1, filters * len(cnn_kernel_size), 3: pos_encode_size]
        conv_pooling = tf.reduce_max(mul,axis=-3)

        # modify by lty
        # 把单卷积核大小，改为多种卷积核大小。
        # old code ####################
        # conv_pooling = tf.reshape(conv_pooling,[-1,sentence_number,self.params.filters*3])
        ###############################

        # conv_pooling = [batch_size, sentence_number, filters * len(cnn_kernel_size) * 3]
        conv_pooling = tf.reshape(conv_pooling,[-1,sentence_number,self.params.filters*3*len(self.params.cnn_kernel_size)])
        conv_pooling = self.dropout(conv_pooling)

        if tf.executing_eagerly():
            print(" conv_pooling \r\n {}\r\n {}".format(conv_pooling,tf.shape(conv_pooling)))
        inputs[self.params.feature_name] = conv_pooling

        return inputs