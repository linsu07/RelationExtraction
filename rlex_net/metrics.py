import tensorflow as tf
from tensorflow.python.estimator.canned.metric_keys import MetricKeys

EPSILON = 1e-7

def cal_f1(precise, recall):
    return 2*precise*recall/(precise+recall+EPSILON)

def get_custom_metrics(num_class):

    # 在原本的方法上，新增了precision recall和f1
    def custom_metrics(labels, class_ids, weights, unreduced_loss, regularization_loss):
        with tf.name_scope(
                None, 'metrics',
                (labels, class_ids, weights, unreduced_loss, regularization_loss)):
            keys = MetricKeys
            metric_ops = {
                # Estimator already adds a metric for loss.
                keys.LOSS_MEAN:
                    tf.metrics.mean(
                        values=unreduced_loss,
                        weights=weights,
                        name=keys.LOSS_MEAN),
                keys.ACCURACY:
                    tf.metrics.accuracy(
                        labels=labels,
                        predictions=class_ids,
                        weights=weights,
                        name=keys.ACCURACY)
            }

            mask = tf.constant([0] + [1] * (int(num_class) - 1), dtype=tf.float32)
            one_hot_labels = tf.squeeze(tf.one_hot(labels, num_class))
            one_hot_predicts = tf.squeeze(tf.one_hot(class_ids, num_class))

            one_hot_labels = tf.multiply(one_hot_labels, tf.transpose(mask))
            one_hot_predicts = tf.multiply(one_hot_predicts, tf.transpose(mask))

            precise_total,precise_op_total = tf.metrics.precision(
                labels=one_hot_labels,
                predictions=one_hot_predicts,
                weights=weights,
                name="precise_total")

            recall_total,recall_op_total = tf.metrics.recall(
                labels=one_hot_labels,
                predictions=one_hot_predicts,
                weights=weights,
                name="recall_total")

            metric_ops[keys.PRECISION] = (precise_total,precise_op_total)
            metric_ops[keys.RECALL] = (recall_total,recall_op_total)

            metric_ops["f1_micro"] = \
                (cal_f1(precise_total, recall_total),cal_f1(precise_op_total,recall_op_total))

            if regularization_loss is not None:
                metric_ops[keys.LOSS_REGULARIZATION] = (
                    tf.metrics.mean(
                        values=regularization_loss,
                        name=keys.LOSS_REGULARIZATION))
        return metric_ops
    return custom_metrics
