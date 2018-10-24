
import tensorflow as tf
from tensorflow.contrib.layers import dense_to_sparse
from tensorflow.contrib.layers.python.ops.sparse_ops import dense_to_sparse_tensor
from ultra.rlex_net.networks.model import model_fn
import collections
#2,2,3
# feature = [
#     [ ["7","7","7","他"],["7","7","7","0"],["8","8","8","0"]],
#     [ ["8","8","8","0"],["8","8","0","0"] ,["0","0","0","0"]]
# ]

feature = [
    [ [1,2,3,7],[7,7,7,0],[8,8,8,0]],
    [ [8,8,8,0],[8,8,0,0] ,[0,0,0,0]]
]
pos = [
    [ [1,1,0,1],[2,0,3,0],[0,0,0,0] ],
    [ [1,1,0,0],[2,0,0,0],[0,0,0,0] ]
]
dis = [
    [ [[1,1],[1,2],[1,4],[1,4]],[[2,1],[1,0],[1,3],[0,0]],[[2,1],[3,5],[2,8],[0,0]] ],
    [ [[1,5],[1,1],[1,1],[0,0]],[[1,1],[1,1],[0,0],[0,0]],[[0,0],[0,0],[0,0],[0,0]]]
]

# label = [[a],[b]]
label = [[1],[2]]

config = collections.namedtuple("config",field_names=[])
config.num_ps_replicas = None
params = collections.namedtuple("params",field_names = [])
params.feature_name = "fea"
params.label_name = "lab"
params.pos_name = "pos"
params.distance_name = "dis"
params.embedding_size = 2
params.embed_size = 2
params.feature_voc_file_len = 5
params.feature_voc_file = "/tmp/test/voc_file.txt"
params.nClasses = 3
params.learning_rate = 0.01
#params.ner_procedure = ["transformer","mi_att2"]
params.ner_procedure = ["pcnn","mi_att"]
params.label_voc_file ="/tmp/test/label_voc_file.txt"
params.label_voc_file_len=3
params.cnn_kernel_size = [2,3,5]
params.filters = 5
params.embedding_file = None
params.voc_length = 3
params.weightColName = "weight"
params.drop_out_rate = 0
params.transfromer_layers = 2
params.transfromer_head_number = 3
params.tansformer_d_model = 24
params.tansformer_shareweight = 0
params.enable_ema = 0
params.weightColName = None
params.rnn_hidden_size = 3


def test():
    fea = tf.constant(feature)
    la= tf.constant(label)
    po = tf.constant(pos)
    features = {
        params.feature_name:dense_to_sparse_tensor(fea),
        params.pos_name:dense_to_sparse_tensor(po),
        params.distance_name:dense_to_sparse_tensor(dis)
    }
    if tf.executing_eagerly():
        print("feature is \r\n {}".format(features[params.feature_name]))
        print("pos is \r\n {}".format(features[params.pos_name]))
    model_fn(features,la,tf.estimator.ModeKeys.TRAIN,config,params)


if __name__=="__main__":
    tf.enable_eager_execution()
    test()
