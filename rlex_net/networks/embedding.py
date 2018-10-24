import tensorflow as tf
from tensorflow.python.ops.lookup_ops import index_table_from_file
from common.word2vec import word_embedding_initializer


'''
  * Created by linsu on 2018/7/26.
  * mailto: lsishere2002@hotmail.com
'''

class PosEmbedLayer(tf.layers.Layer):
    def __init__(self,params,dtype = tf.float32,name = "pos_embedding"):
        super(PosEmbedLayer,self).__init__(True,name,dtype)
        self.params = params
    def build(self, _):

        self.pos_embedding = tf.constant([
            [0.0, 0.0, 0.0]   #in
            , [1.0, 0.0, 0.0]  #before
            , [0.0, 1.0, 0.0]  #between
            , [0.0, 0.0, 1.0]  #after
        ], dtype=tf.float32, name="pos_embedding")
        self.built = True

    def call(self, inputs, **kwargs):

        pos_ids = inputs[self.params.pos_name]
        if isinstance(pos_ids, tf.SparseTensor):
            pos_ids = tf.sparse_tensor_to_dense(pos_ids,default_value=0,name = "sparsePosId2dense")
        if tf.executing_eagerly():
            print("pos ids is \r\n {}".format(pos_ids))
        pos_embedding = tf.nn.embedding_lookup(self.pos_embedding,pos_ids)
        if tf.executing_eagerly():
            print("pos embedding is \r\n {}".format(pos_embedding))
        inputs[self.params.pos_name] = pos_embedding
        return inputs

class DistanceEmbedding(tf.layers.Layer):
    def __init__(self,params,dtype=tf.float32,name = "distance_embedding"):
        super(DistanceEmbedding,self).__init__(True,name,dtype)
        self.params = params

    def build(self, _):

        zero = self.add_variable(
            name ="1_pos"
            ,shape = [1,5]
            ,initializer=tf.zeros_initializer()
            ,trainable=False
             )
        self.distance1 = tf.concat([zero,self.add_variable(
            name = "distance1"
            ,shape = [10000,5]
            ,initializer = tf.random_uniform_initializer(-1, 1)
        )],0)
        self.distance2 = tf.concat([zero,self.add_variable(
            name = "distance2"
            ,shape = [10000,5]
            ,initializer = tf.random_uniform_initializer(-1, 1)
        )],0)
        self.built = True

    def call(self, inputs, **kwargs):
        distance_feature = inputs[self.params.distance_name]
        try:
            distance_feature = tf.sparse_tensor_to_dense(distance_feature)
        except:
            print("Input must be a SparseTensor.")
        distance_pos = distance_feature.get_shape().as_list()
        shape = tf.shape(distance_feature)

        # b_last = distance_pos[-1]
        # distance_pos[-1] = tf.div(b_last, 2)
        # distance_pos.append(2)
        x = tf.gather(shape, 0)
        y = tf.gather(shape, 1)
        z = tf.gather(shape, 2)
        distance_feature = tf.reshape(distance_feature, [x,y,-1,2])
        distance_feature = tf.floormod(distance_feature,10000)
        [d1,d2] = tf.split(distance_feature,2,axis= -1)
        d1 = tf.nn.embedding_lookup(self.distance1,tf.squeeze(d1,-1))
        d2 = tf.nn.embedding_lookup(self.distance2,tf.squeeze(d2,-1))

        last_feature = inputs[self.params.feature_name]
        inputs[self.params.feature_name] = tf.concat([last_feature,d1,d2],axis = -1)
        return inputs


class WordEmbedLayer(tf.layers.Layer):
    def __init__(self, params, dtype=tf.float32, name="word_embedding"):
        super(WordEmbedLayer, self).__init__(True, name, dtype)
        self.params = params
        self.num_oov_buckets = 2000

    def build(self, _):
        padding = self.add_variable(
            name="padding"
            , shape=[1, self.params.embedding_size]
            , initializer=tf.zeros_initializer()
            ,trainable=False
        )
        unk = self.add_variable(
            name="unk"
            , shape=[1, self.params.embedding_size]
            , initializer=tf.random_uniform_initializer(-1, 1)
        )

        if not self.params.embedding_file:
            embedding_other = self.add_variable(
                name="embedding_other"
                , shape=[self.params.feature_voc_file_len - 2, self.params.embedding_size]
                , initializer=tf.random_uniform_initializer(-1, 1)
            )

        else:
            embedding_other = self.add_variable(
                name="embedding_other"
                , shape=[self.params.feature_voc_file_len - 2, self.params.embedding_size]
                , initializer=word_embedding_initializer(self.params.embedding_file, include_word=False, vector_length=self.params.embedding_size)
            )
        tf.add_to_collection("not_in_ema",embedding_other)
        embedding_oov = self.add_variable(
                name="embedding_oov"
                , shape=[self.num_oov_buckets, self.params.embedding_size]
                , initializer=tf.random_uniform_initializer(-1, 1)
        )
        tf.add_to_collection("not_in_ema",embedding_oov)
        self.embedding = tf.concat([padding, unk, embedding_other,embedding_oov], axis=0)
        if tf.executing_eagerly():
            tmp = tf.slice(self.embedding,[0,0],[4,-1])
            print("embedding is \r\n {}".format(tmp))

        # self.feature_lookup_table = index_table_from_file(
        #     vocabulary_file=self.params.feature_voc_file,
        #     num_oov_buckets=self.num_oov_buckets,
        #     vocab_size=self.params.feature_voc_file_len,
        #     default_value=1,
        #     key_dtype=tf.string,
        #     name='feature_index_lookup')
        self.built = True

    def call(self, inputs, **kwargs):
        # sentences_ids = self.feature_lookup_table.lookup(inputs[self.params.feature_name])
        sentences_ids = inputs[self.params.feature_name]
        if tf.executing_eagerly():
            print("sentences_ids\r\n {}".format(sentences_ids))
        if isinstance(sentences_ids, tf.SparseTensor):
            sentences_ids = tf.sparse_tensor_to_dense(sentences_ids,default_value=0,name = "sparseid2dense")
        mask = tf.cast(tf.abs(tf.sign(sentences_ids)),tf.float32)
        inputs["_mask"] = mask
        sen_len = tf.reduce_sum(mask, -1)
        inputs["sen_length"] = sen_len
        sentences_embedding = tf.nn.embedding_lookup(self.embedding,sentences_ids)
        inputs[self.params.feature_name] = sentences_embedding
        if tf.executing_eagerly():
            print("sentences_embedding is \r\n {}".format(sentences_embedding))
        return inputs


