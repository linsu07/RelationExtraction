import math

import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

from rlex_net.parameters import user_params

'''
  * Created by linsu on 2018/9/4.
  * mailto: lsishere2002@hotmail.com
'''

class MIAttentionLayer2(tf.layers.Layer):
    def __init__(self, params: user_params, feature_size:int,is_trainning=True, name="mi_attention_layer2", dtype=tf.float32):
        super(MIAttentionLayer2, self).__init__(is_trainning, name, dtype)
        self.params = params
        self.feature_size = feature_size
        self.exp_epsilon = -1e20

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
        self.dropout = tf.layers.Dropout(rate=self.params.drop_out_rate, name="dropout", trainable=self.trainable)
        self.built = True

    def attention_train(self, docs, label_id,mask):
        if tf.executing_eagerly():
            print("in attentionProject docs is \r\n {} \r\n {}".format(docs,tf.shape(docs)))
            print("in attentionProject label_ids is \r\n {} \r\n {}".format(label_id,tf.shape(label_id)))

        # querys = [batch_size, 1, feature_size]
        querys = tf.nn.embedding_lookup(self.weight_transpose,label_id)
        # batch_size = tf.shape(docs)[0]
        # sent_num = tf.shape(docs)[1]
        # seq_len = tf.shape(docs)[2]
        #docs [batch_size,sent_num,seq_len_feature_size]
        #docs_reshape = tf.reshape(docs,[-1,sent_num*seq_len,self.feature_size])

        #[batch_size,1,1,self.feature_size]
        querys_expand = tf.expand_dims(querys,1)
        #[-1,sent_num,seq_len,1]
        factor_seq = tf.reduce_sum(tf.multiply(docs,querys_expand),axis=-1,keep_dims=True)
        #[-1,sent_num,seq_len]
        #factor_seq = tf.reshape(tf.matmul(docs_reshape,querys,transpose_b=True),[-1,sent_num,seq_len,1])
        mask_expand = tf.expand_dims(mask,-1)
        mask = (1-mask_expand)*self.exp_epsilon
        factor_seq = factor_seq*self.scale+mask
        #factor_seq = tf.where(tf.cast(factor_seq,tf.bool),factor_seq,tf.ones_like(factor_seq)*self.exp_epsilon)
        #[-1,sent_num,seq_len,1]
        factor_seq = tf.nn.softmax(factor_seq,-2)
        factor_seq = factor_seq*mask_expand
        #[batch_size,sent_num,1,feature_size]
        att_feature = tf.matmul(factor_seq,docs,transpose_a=True)
        #[batch_size,sent_num,feature_size]
        att_feature = tf.squeeze(att_feature,-2)

        docs = att_feature

        factor = tf.multiply(docs,querys)
        if tf.executing_eagerly():
            print("in attentionProject factor is \r\n {} \r\n {}".format(factor,tf.shape(factor)))

        # 对multi-instance包里每句话feature_size个特征求和并乘以一个因子scale
        factor = tf.multiply(tf.reduce_sum(factor,axis = -1),self.scale)
        # factor = [batch_size, sentence_number]

        factor = tf.where(tf.cast(factor,tf.bool),x= factor,y = tf.ones_like(factor)* self.exp_epsilon)
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
            print("train logits is:")
            print (logits)
        return logits



    # def attention_predict(self,docs,mask):
    #     if tf.executing_eagerly():
    #         print("in attentionProject docs is \r\n {} \r\n {}".format(docs,tf.shape(docs)))
    #     #docs [batch_size,sent_num,seq_len_feature_size]
    #     batch_size = tf.shape(docs)[0]
    #     sent_num = tf.shape(docs)[1]
    #     seq_len = tf.shape(docs)[2]
    #     docs_reshape = tf.reshape(docs,[batch_size,sent_num*seq_len,self.feature_size])
    #     #[batch_size,feature_size, nclasses]
    #     weigth = tf.tile(tf.expand_dims(self.weight,0),[batch_size,1,1])
    #     #[batch_size,,sent_num,seq_len,nclasses]
    #     factor =tf.reshape( tf.matmul(docs_reshape,weigth),[batch_size,sent_num,seq_len,self.params.nClasses])*self.scale
    #     factor = tf.where(tf.cast(factor,tf.bool),factor,tf.ones_like(factor)*self.exp_epsilon)
    #     factor = tf.nn.softmax(factor,-2)
    #     #[batch_size,,sent_num,nlasses,feature_size]
    #     feature = tf.matmul(tf.transpose(factor,[0,1,3,2]),docs)
    #
    #     weitht2 =tf.reshape(self.weight_transpose,[1,1,self.params.nClasses,self.feature_size])
    #     #[batch_size,sent_num,nlasses]
    #     factor = tf.reduce_sum(tf.multiply(feature,weitht2),-1)*self.scale
    #     factor = tf.where(tf.cast(factor,tf.bool),factor,tf.ones_like(factor)*self.exp_epsilon)
    #     factor = tf.nn.softmax(factor,-2)
    #     #[batch_size,sent_num,nlasses,1]
    #     factor = tf.expand_dims(factor,-1)
    #     #[batch_size,nclasses,feature_size]
    #     feature = tf.reduce_sum(tf.multiply(factor,feature),1)
    #     #[batch_size,nclasses,nclasses]
    #     logits = tf.add(tf.matmul(feature,weigth),self.bias)
    #     if tf.executing_eagerly():
    #         print("raw predict logits is \r\n {} \r\n {}".format(logits,tf.shape(logits)))
    #     #[batch_size,nclasses,nclasses]
    #     logits = tf.nn.softmax(logits,-1)
    #     if tf.executing_eagerly():
    #         print("in attentionProject logits is \r\n {} \r\n {}".format(logits,tf.shape(logits)))
    #     #logits = tf.reduce_sum(tf.multiply(logits,self.diag_mask),-1)
    #     logits = tf.matrix_diag_part(logits)
    #     if tf.executing_eagerly():
    #         print("predict logits is \r\n {} \r\n {}".format(logits,tf.shape(logits)))
    #     return logits

    def attention_predict(self,docs,mask):
        if tf.executing_eagerly():
            print("in attentionProject docs is \r\n {} \r\n {}".format(docs,tf.shape(docs)))
        #docs [batch_size,sent_num,seq_len_feature_size]
        batch_size = tf.shape(docs)[0]
        sent_num = tf.shape(docs)[1]
        seq_len = tf.shape(docs)[2]
        docs_reshape = tf.reshape(docs,[batch_size,sent_num,seq_len,1,self.feature_size])
        weight = tf.reshape(self.weight_transpose,[1,1,1,self.params.nClasses,self.feature_size])
        #[batch_size,,sent_num,seq_len,nclasses]
        factor = tf.reduce_sum(tf.multiply(docs_reshape,weight),-1)*self.scale

        #factor =tf.reshape( tf.matmul(docs_reshape,weight),[batch_size,sent_num,seq_len,self.params.nClasses])*self.scale
        factor = tf.where(tf.cast(factor,tf.bool),factor,tf.ones_like(factor)*self.exp_epsilon)
        factor = tf.nn.softmax(factor,-2)
        #[batch_size,,sent_num,nlasses,feature_size]
        feature = tf.matmul(tf.transpose(factor,[0,1,3,2]),docs)

        #[1,1,nClasses,self.feature_size]
        weight = tf.squeeze(weight,0)

        #[batch_size,sent_num,nclasses]
        factor = tf.reduce_sum(tf.multiply(feature,weight),-1)*self.scale
        factor = tf.where(tf.cast(factor,tf.bool),factor,tf.ones_like(factor)*self.exp_epsilon)

        factor = tf.nn.softmax(factor,-2)
        #[batch_size,sent_num,nlasses,1]
        factor = tf.expand_dims(factor,-1)
        #[batch_size,nclasses,feature_size]
        feature = tf.reduce_sum(tf.multiply(factor,feature),1)
        if tf.executing_eagerly():
            print("raw predict feature is \r\n {} \r\n {}".format(feature,tf.shape(feature)))

        #[batch_size,feature_size,nclasses]
        weight2 = tf.tile(tf.expand_dims(self.weight,0),[batch_size,1,1])
        #[batch_size,nclasses,nclasses]
        logits = tf.add(tf.matmul(feature,weight2),self.bias)
        if tf.executing_eagerly():
            print("raw predict logits is \r\n {} \r\n {}".format(logits,tf.shape(logits)))
        #[batch_size,nclasses,nclasses]
        logits = tf.nn.softmax(logits,-1)
        if tf.executing_eagerly():
            print("in attentionProject logits is \r\n {} \r\n {}".format(logits,tf.shape(logits)))
        #logits = tf.reduce_sum(tf.multiply(logits,self.diag_mask),-1)
        logits = tf.matrix_diag_part(logits)
        if tf.executing_eagerly():
            print("predict logits is \r\n {} \r\n {}".format(logits,tf.shape(logits)))
        return logits

    def call(self, inputs, **kwargs):
        docs = inputs[self.params.feature_name]
        if self.trainable:
            label_id = inputs[self.params.label_name]
            if tf.executing_eagerly():
                print("label id is \r\n {} \r\n {}".format(label_id,tf.shape(label_id)))
            logits1 = self.attention_train(docs,label_id,inputs["_mask"])
            return logits1
        else:
            logits = self.attention_predict(docs,inputs["_mask"])
            return logits


