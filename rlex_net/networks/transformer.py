import tensorflow as tf
from ultra.common.networks.attention_is_all import GoogleTransformer


class Transformer(tf.layers.Layer):
    def __init__(self,params,feature_size, isTraining = True, name="Transformer",dtype=tf.float32):
        super(Transformer, self).__init__( name=name,trainable=isTraining)
        self.params = params
        self.feature_size = feature_size

    def build(self, _):
        self.layers = []
        if self.params.tansformer_shareweight==0:
            for i in range(self.params.transfromer_layers):
                self.layers.append(GoogleTransformer(self.params.transfromer_head_number,self.params.tansformer_d_model
                                                     ,self.trainable,name ="transformer_{}".format(i)
                                                     ,dropout_rate = self.params.drop_out_rate ))

        else:
            layer = GoogleTransformer(self.params.transfromer_head_number,self.params.tansformer_d_model
                                                        ,self.trainable,name ="transformer"
                                                        ,dropout_rate = self.params.drop_out_rate )
            for i in range(self.params.transfromer_layers):
                self.layers.append(layer)

    def call(self, inputs, **kwargs):
        feature = inputs[self.params.feature_name]
        shape = tf.shape(feature)
        batch_size,sent_num,seq_len = shape[0],shape[1],shape[2]
        feature = tf.reshape(feature,[batch_size*sent_num,seq_len,self.feature_size])

        mask = tf.reshape(inputs["_mask"],[batch_size*sent_num,seq_len])
        for layer in self.layers:
            feature = layer(feature,mask =mask )
        inputs[self.params.feature_name] = tf.reshape(feature,[batch_size,sent_num,seq_len,self.params.tansformer_d_model])

        return inputs