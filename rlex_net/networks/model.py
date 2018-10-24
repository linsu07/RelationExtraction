import tensorflow as tf
from tensorflow.contrib.estimator import multi_class_head#, multi_label_head
from ultra.classification.common.head import multi_label_head
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.python.estimator.run_config import RunConfig
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.training import training_util

from ultra.common.networks.dimension_size_control import DimensionSizeControl
from ultra.rlex_net.networks.AttentionProject import MIAttentionLayer
from ultra.rlex_net.networks.AttentionProject2 import MIAttentionLayer2
from ultra.rlex_net.networks.birnn import BIRnnLayer
from ultra.rlex_net.networks.birnn2 import BIRnnLayer2
from ultra.rlex_net.networks.embedding import PosEmbedLayer, WordEmbedLayer, DistanceEmbedding
from ultra.rlex_net.networks.highway_network import Highway
from ultra.rlex_net.networks.pcnn import PcnnLayer
from ultra.rlex_net.metrics import get_custom_metrics
from ultra.rlex_net.networks.transformer import Transformer
from ultra.rlex_net.parameters import user_params
'''
  * Created by linsu on 2018/7/26.
  * mailto: lsishere2002@hotmail.com
'''

def model_fn(features,labels,mode:tf.estimator.ModeKeys,config: RunConfig, params:user_params):
    print("--- model_fn in %s ---" % mode)
    #ema = tf.train.ExponentialMovingAverage(decay=0.99)
    num_ps_replicas = config.num_ps_replicas if config else 0
    if tf.executing_eagerly():
        partitioner = None
    else:
        partitioner = partitioned_variables.min_max_variable_partitioner(
            max_partitions=num_ps_replicas)

        partitioner = None

    # def custom_getter(getter,name, *args, **kwargs):  # noqa
    #     var = getter(name, *args, **kwargs)
    #     var_name = ema.average_name(var)
    #     av_var = getter(var_name,*args,**kwargs)
    #     print("avg var name = {}, value {}".format(var_name,av_var))
    #     return av_var if av_var else var
    layers = list()
    with tf.variable_scope("rlex", partitioner=partitioner, initializer=xavier_initializer()
                           ,custom_getter= None) as scope:
        layers.append(WordEmbedLayer(params))
        feature_size = 0
        for name in params.ner_procedure:
            if name == "pcnn":
                layers.append(PosEmbedLayer(params))
                layers.append(PcnnLayer(params,is_trainning=(mode==tf.estimator.ModeKeys.TRAIN)))
                feature_size = params.filters*3*len(params.cnn_kernel_size)
            elif name == "mi_att":
                layers.append(MIAttentionLayer(params,feature_size=feature_size,is_trainning=(mode==tf.estimator.ModeKeys.TRAIN)))
            elif name == "mi_att2":
                layers.append(MIAttentionLayer2(params,feature_size=feature_size,is_trainning=(mode==tf.estimator.ModeKeys.TRAIN)))

            elif name == "birnn":
                layers.append(DistanceEmbedding(params))
                feature_size = params.embed_size+2* 5
                layers.append(BIRnnLayer(params,feature_size,is_trainning=(mode ==tf.estimator.ModeKeys.TRAIN)))
                feature_size = 2*params.rnn_hidden_size
            elif name == "highway":
                layers.append(Highway(params, output_size=feature_size, trainable=(mode==tf.estimator.ModeKeys.TRAIN)))

            elif name=="transformer":
                layers.append(DistanceEmbedding(params))
                layers.append(DimensionSizeControl(axis=1, max_size=100, control_tensor_names=[params.feature_name, "sen_length", "_mask"]))
                feature_size = params.embedding_size+2* 5
                layers.append(Transformer(params,feature_size,mode==tf.estimator.ModeKeys.TRAIN))
                feature_size = params.tansformer_d_model
            elif name == "birnn2":
                layers.append(DistanceEmbedding(params))
                feature_size = params.embed_size+2* 5
                layers.append(BIRnnLayer2(params,feature_size,is_trainning=(mode ==tf.estimator.ModeKeys.TRAIN)))
                feature_size = 2*params.rnn_hidden_size
            else:
                raise ValueError("unknow precedure, valid name is pcnn,mi_att,birnn,highway")

        head = multi_label_head(params.nClasses, weight_column=params.weightColName, thresholds=[0.3, 0.5, 0.6, 0.7]) \
            if params.is_multi_label \
            else multi_class_head(params.nClasses,label_vocabulary=None, weight_column=params.weightColName)

        if mode==tf.estimator.ModeKeys.TRAIN:
            features[params.label_name] = labels

        logits = features
        for layer in layers:
            logits = layer(logits)

    def train_op_fn(loss):
        global_step=training_util.get_global_step()
        lr = tf.minimum(params.learning_rate, 0.001 / tf.log(999.) * tf.log(tf.cast(global_step, tf.float32) + 1))
        opt = tf.train.AdamOptimizer(learning_rate = lr, beta1 = 0.8, beta2 = 0.999, epsilon = 1e-7)
        grads = opt.compute_gradients(loss)
        gradients, variables = zip(*grads)
        capped_grads, _ = tf.clip_by_global_norm(
            gradients, 5)
        train_op = opt.apply_gradients(
            zip(capped_grads, variables), global_step=global_step)
        return train_op
        # train_op = tf.train.AdamOptimizer(learning_rate=params.learning_rate) \
        #     .minimize(loss, global_step=training_util.get_global_step())
        # return train_op

    if params.enable_ema and mode==tf.estimator.ModeKeys.TRAIN:
        ema = tf.train.ExponentialMovingAverage(decay=0.9)
        trained_var = tf.trainable_variables()
        ema_op = ema.apply(trained_var)
        # for var in tf.get_collection(key = "not_in_ema"):
        #     trained_var.remove(var)
        variables_to_restore = ema.variables_to_restore()
        #print( "in train-------------------------------------")
        #print(variables_to_restore)
        with tf.control_dependencies([ema_op]):
            logits = tf.identity(logits)
            #print("in ema")

    # 替换了head自带的评估方法，补充了几个评估指标
    if not params.is_multi_label:
        head._eval_metric_ops = get_custom_metrics(num_class=params.nClasses)

    # regularization_loss = [0.001 * x for x in tf.losses.get_regularization_losses()]
    spec = head.create_estimator_spec(
        features, mode, logits, labels=labels, train_op_fn=train_op_fn)
    # features, mode, logits, labels=labels, train_op_fn=train_op_fn, regularization_losses=regularization_loss)

    if params.enable_ema and mode==tf.estimator.ModeKeys.PREDICT:
        ema = tf.train.ExponentialMovingAverage(decay=0.99)
        variables_to_restore = ema.variables_to_restore()
        #print( "in prodict-------------------------------------")
        #print(variables_to_restore)
        scaffold = spec.scaffold
        scaffold._saver =tf.train.Saver(variables_to_restore)

    return spec


