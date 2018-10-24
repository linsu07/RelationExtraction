import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from ultra.rlex_net.parameters import user_params

class Highway(tf.layers.Layer):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).
    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """
    def __init__(self, params: user_params, output_size, num_layers = 1, bias = -2.0, activation_fun = tf.nn.relu, trainable=True, name=None, dtype=None,
                                                       activity_regularizer=None, **kwargs):

        super(Highway, self).__init__(trainable=trainable, name=name,
                                    activity_regularizer=activity_regularizer,
                                    **kwargs)

        self.params = params
        self.output_size = output_size
        self.num_layers = num_layers
        self.bias = bias
        self.activation_fun = activation_fun
        self.layers = []


    def build(self, input_shape):
        for idx in range(self.num_layers):
            dense_net_1 = tf.layers.Dense(self.output_size, kernel_initializer=xavier_initializer(), activation=tf.nn.relu, kernel_regularizer=tf.nn.l2_loss)
            #self.add_loss(dense_net_1.losses)
            dense_net_2 = tf.layers.Dense(self.output_size, kernel_initializer=xavier_initializer(), activation=tf.nn.relu, kernel_regularizer=tf.nn.l2_loss)
            #self.add_loss(dense_net_2.losses)
            self.layers.append((dense_net_1, dense_net_2))
        self.built = True

    def call(self, inputs, *args, **kwargs):
        print("Highway.call()")
        docs = inputs[self.params.feature_name]
        sequence_num = tf.shape(docs)[1]
        docs = tf.reshape(docs, [-1, self.output_size])
        output = docs
        for idx in range(self.num_layers):
            output_linear = self.layers[idx][0](docs)
            g = self.activation_fun(output_linear)
            Wy = self.layers[idx][1](docs)
            t = tf.sigmoid(Wy + self.bias)
            output = t * g + (1. - t) * docs
            docs = output

        inputs[self.params.feature_name] = tf.reshape(output, [-1, sequence_num, self.output_size])
        return inputs

    def __call__(self, inputs, *args, **kwargs):
        return super(Highway, self).__call__(inputs, *args, **kwargs)

def highway(inputs, params: user_params, output_size, num_layers = 1, bias = -2.0, activation_fun = tf.nn.relu, trainable=True, name=None, dtype=None,
            activity_regularizer=None, **kwargs):
    layer = Highway(params, output_size, num_layers, bias, activation_fun, trainable, name, dtype,
                   activity_regularizer, **kwargs)


    return layer(inputs)