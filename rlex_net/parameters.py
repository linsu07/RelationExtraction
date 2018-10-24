import collections
import tensorflow as tf
import os
import numpy as np


class user_params(collections.namedtuple("namedtuple",
                                         ["ner_procedure","label_voc_file_path"
                                         ,"label_name","learning_rate"
                                         ,"pos_name","embed_size",
                                          "embedding_file_path",
                                          "feature_name"
                                          ,"cnn_kernel_size","filters",
                                          'rnn_hidden_size',"distance_name",
                                          "data_dir", "model_dir", "batch_size",
                                          "feature_voc_file_path", "drop_out_rate"
                                          ,"enable_ema"
                                          ,"transfromer_layers","transfromer_head_number","tansformer_d_model"
                                          ,"tansformer_shareweight"
                                          ,"gpu_cors", "num_classes", "num_features", "is_multi_label"])):
    pass

def enrich_hyper_parameters(params: user_params):

    # feature
    # if params.embedding_file_path:
    #     suggest_file_name = os.path.join(params.embedding_file_path, "vocabulary")
    # else:
    #     suggest_file_name = os.path.join(params.feature_voc_file_path, "vocabulary")
    # if not tf.gfile.Exists(suggest_file_name):
    #     files = tf.gfile.ListDirectory(params.feature_voc_file_path)
    #     vocab_files = [file for file in files if file.endswith(".txt")]
    #     vocab_file = os.path.join(params.feature_voc_file_path, vocab_files[0])
    #     tf.gfile.Copy(vocab_file, suggest_file_name)
    # params.feature_voc_file = suggest_file_name
    # params.feature_voc_file_len = get_vocab_file_size(suggest_file_name)
    params.feature_voc_file_len = params.num_features

    # label
    # suggest_file_name = os.path.join(params.label_voc_file_path, "label_vocabulary")
    # if not tf.gfile.Exists(suggest_file_name):
    #     files = tf.gfile.ListDirectory(params.label_voc_file_path)
    #     vocab_files = [file for file in files if file.endswith(".txt")]
    #     vocab_file = os.path.join(params.label_voc_file_path, vocab_files[0])
    #     tf.gfile.Copy(vocab_file, suggest_file_name)
    # params.label_voc_file = suggest_file_name
    # params.nClasses = params.nClasses
    params.nClasses = params.num_classes

    # embedding
    if params.embedding_file_path:
        params.embedding_file = os.path.join(params.embedding_file_path, "embedding")
        np_array = np.loadtxt(tf.gfile.GFile(params.embedding_file, "r"))
        params.embedding_size = np_array.shape[1]
    else:
        params.embedding_file = None
        params.embedding_size = params.embed_size

    # 加一个weight 列
    params.weightColName = "weight"

    check_params(params)


def check_params(params: user_params):
    if params.tansformer_d_model % params.transfromer_head_number != 0:
        raise ValueError("参数不合理: tansformer_d_model 必须是 transfromer_head_number 的整数倍")