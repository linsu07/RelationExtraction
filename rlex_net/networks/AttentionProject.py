import math

import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

from rlex_net.parameters import user_params

'''
  * Created by linsu on 2018/7/27.
  * mailto: lsishere2002@hotmail.com
'''

class MIAttentionLayer(tf.layers.Layer):
    def __init__(self, params: user_params, feature_size:int,is_trainning=True, name="mi_attention_layer", dtype=tf.float32):
        super(MIAttentionLayer, self).__init__(is_trainning, name, dtype)
        self.params = params
        self.feature_size = feature_size

    def build(self, _):
        # weight = [feature_size, nClasses]
        self.weight = self.add_variable(
            name = "weights",
            shape = [self.feature_size,self.params.nClasses],
            initializer=xavier_initializer(),
            regularizer=tf.nn.l2_loss
        )
        # weight_transpose = [nClasses, feature_size]
        self.weight_transpose = tf.transpose(self.weight)
        self.bias = self.add_variable(
            name = "bias",
            shape=[self.params.nClasses],
            initializer=tf.zeros_initializer
        )
        self.scale = 1 / math.sqrt(self.feature_size)
        index = tf.constant([i for i in range(self.params.nClasses)])
        print("in attentionProject diag_mask is \r\n {} ".format(index))
        self.diag_mask = tf.expand_dims(tf.one_hot(index,depth=self.params.nClasses,dtype=tf.float32),0)
        if tf.executing_eagerly():
            print("in attentionProject diag_mask is \r\n {} \r\n {}".format(self.diag_mask,tf.shape(self.diag_mask)))
        self.dropout = tf.layers.Dropout(rate=self.params.drop_out_rate, name="dropout", trainable=self.trainable)
        self.built = True

    def attention_train(self, docs, label_id):
        if tf.executing_eagerly():
            print("in attentionProject docs is \r\n {} \r\n {}".format(docs,tf.shape(docs)))
            print("in attentionProject label_ids is \r\n {} \r\n {}".format(label_id,tf.shape(label_id)))
        # querys是label_id的embedding，每个label_id对应一个feature_size长度的权重向量
        # querys = [batch_size, 1, feature_size]
        querys = tf.nn.embedding_lookup(self.weight_transpose,label_id)
        # factor是multi-instance包里每句话的feature_size个特征与querys的乘积。
        # 得到每句话的特征加权后的特征表达，共feature_size个特征。
        # docs = [batch_size, sentence_number, feature_size]
        # factor = [batch_size, sentence_number, feature_size]
        factor = tf.multiply(docs,querys)
        if tf.executing_eagerly():
            print("in attentionProject factor is \r\n {} \r\n {}".format(factor,tf.shape(factor)))

        # 对multi-instance包里每句话feature_size个特征求和并乘以一个因子scale
        factor = tf.multiply(tf.reduce_sum(factor,axis = -1),self.scale)
        # factor = [batch_size, sentence_number]
        padding = -1e15
        factor = tf.where(tf.cast(factor,tf.bool),x= factor,y = tf.ones_like(factor)* padding)
        # 经过softmax后得到multi-instance包里每句话对分类的贡献概率分布。
        # factor = [batch_size, sentence_number, 1]
        factor = tf.expand_dims(tf.nn.softmax(factor,axis = -1),axis=-1)
        if tf.executing_eagerly():
            print("in attentionProject factor after softmax is \r\n {} \r\n {}".format(factor,tf.shape(factor)))
        # new_feature是在原始docs基础上，对multi-instance包里每句话的特征，加了句子级的权重。
        # 因为每个句子，对relation的识别的贡献是不一样的。
        # new_feature = [batch_size, sentence_number, feature_size]
        new_feature = tf.multiply(docs,factor)
        # 在multi-instance包里每句话有了不同权重后，加合在一起
        # 这样multi-instance包无论有多少个句子，最终都被表达成feature_size长度的向量
        # new_feature = [batch_size, feature_size]
        new_feature = tf.reduce_sum(new_feature,axis=-2)
        new_feature = self.dropout(new_feature)
        if tf.executing_eagerly():
            print("in attentionProject new_feature after softmax is \r\n {} \r\n {}".format(new_feature,tf.shape(new_feature)))
        # 常规的全连接层
        # logits = [batch_size, n_class]
        logits = tf.add(tf.matmul(new_feature, self.weight), self.bias)
        if tf.executing_eagerly():
            print("in train logits is \r\n {} \r\n {}".format(logits,tf.shape(logits)))

        return logits



    def attention_predict(self,docs):
        # def attention(doc,w_list):
        #     mul = tf.matmul(doc,w_list)
        #     factor = tf.nn.softmax(mul,axis = -2)
        #     doc = tf.expand_dims(doc,-2)
        #     factor = tf.expand_dims(factor,-1)
        #     attention_feature = tf.reduce_sum(tf.multiply(doc,factor),axis=-3)
        #
        #     logit_raw = tf.nn.softmax(tf.add(tf.matmul(attention_feature,self.weight),self.bias),-1)
        #     logit = tf.diag_part(logit_raw)
        #     return logit
        # logits_list = [attention(doc,w_list) for doc in doc_list]
        # return tf.stack(logits_list,axis=0)
        if tf.executing_eagerly():
            print("in attentionProject docs is \r\n {} \r\n {}".format(docs,tf.shape(docs)))
        #[2,2,3]
        sentence_number = tf.shape(docs)[1]
        docs_reshape = tf.reshape(docs,[-1,self.feature_size])
        #[batch_size*sentence_nunmber,classes]
        factor = tf.matmul(docs_reshape,self.weight)
        if tf.executing_eagerly():
            print("in attentionProject factor is \r\n {} \r\n {}".format(factor,tf.shape(factor)))
        factor = tf.multiply(factor,self.scale)
        #[batch_size,sentence_nunmber,classes]
        factor = tf.reshape(factor,[-1,sentence_number,self.params.nClasses])
        if tf.executing_eagerly():
            print("in attentionProject factor is \r\n {} \r\n {}".format(factor,tf.shape(factor)))
        padding = -1e15
        factor = tf.where(tf.cast(factor,tf.bool),x= factor,y = tf.ones_like(factor)* padding)

        factor = tf.nn.softmax(factor,-2)
        if tf.executing_eagerly():
            print("in attentionProject factor is \r\n {} \r\n {}".format(factor,tf.shape(factor)))
        factor = tf.expand_dims(factor,axis=-1)
        if tf.executing_eagerly():
            print("in attentionProject factor is \r\n {} \r\n {}".format(factor,tf.shape(factor)))
        docs = tf.expand_dims(docs,axis = -2)
        logits_raw = tf.multiply(docs,factor)
        if tf.executing_eagerly():
            print("in attentionProject tf.multiply(docs,factor) is \r\n {} \r\n {}".format(logits_raw,tf.shape(logits_raw)))
        logits_raw = tf.reshape(tf.reduce_sum(logits_raw,axis=-3),[-1,self.feature_size])
        logits = tf.add(tf.matmul(logits_raw,self.weight),self.bias)
        if tf.executing_eagerly():
            print("in attentionProject raw logits is \r\n {} \r\n {}".format(logits,tf.shape(logits)))
        #logits = tf.nn.softmax(logits,-1)
        logits = tf.reshape(logits,[-1,self.params.nClasses,self.params.nClasses])
        if tf.executing_eagerly():
            print("in attentionProject logits is \r\n {} \r\n {}".format(logits,tf.shape(logits)))
        #logits = tf.reduce_sum(tf.multiply(logits,self.diag_mask),-1)
        logits = tf.matrix_diag_part(logits)
        if tf.executing_eagerly():
            print("in attentionProject logits is \r\n {} \r\n {}".format(logits,tf.shape(logits)))
        return logits

    def call(self, inputs, **kwargs):
        if self.params.is_multi_label:
            docs = inputs[self.params.feature_name]
            logits = self.attention_predict(docs)
            return logits
        else:
            docs = inputs[self.params.feature_name]
            if self.trainable:
                label_id = inputs[self.params.label_name]
                if tf.executing_eagerly():
                    print("label id is \r\n {} \r\n {}".format(label_id,tf.shape(label_id)))
                logits = self.attention_train(docs,label_id)
                return logits
            else:
                logits = self.attention_predict(docs)
                return logits


