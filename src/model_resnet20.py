from __future__ import division

import ipdb
import math
import tensorflow as tf
import numpy as np

import model_utils


FLAGS = tf.app.flags.FLAGS


def one_hot_embedding(label, n_classes):
    """
    One-hot embedding
    Args:
        label: int32 tensor [B]
        n_classes: int32, number of classes
    Return:
        embedding: tensor [B x n_classes]
    """
    embedding_params = np.eye(n_classes, dtype=np.float32)
    with tf.device('/cpu:0'):
        params = tf.constant(embedding_params)
        embedding = tf.gather(params, label)
    return embedding


def conv2d(x, n_in, n_out, k, s, p='SAME', bias=False, scope='conv'):
    bias = True # TESTME
    with tf.variable_scope(scope):
        kernel = tf.Variable(
            tf.truncated_normal([k, k, n_in, n_out],
                stddev=math.sqrt(2/(k*k*n_out))),
            name='weight')
        weight_decay = tf.mul(tf.nn.l2_loss(kernel), FLAGS.weight_decay, 'weight_decay_loss')
        tf.add_to_collection('losses', weight_decay)
        conv = tf.nn.conv2d(x, kernel, [1,s,s,1], padding=p)
        if bias:
            bias = tf.Variable(tf.zeros([n_out]), name='bias')
            conv = tf.nn.bias_add(conv, bias)
    return conv


def batch_norm(x, n_out, scope='bn', affine=True):
    def mean_var(x, axes, name=None):
        divisor = 1
        for d in xrange(len(x.get_shape())):
            if d in axes:
                divisor *= tf.shape(x)[d]
        divisor = 1.0 / tf.cast(divisor, tf.float32)
        mean = tf.mul(tf.reduce_sum(x, axes), divisor)
        var = tf.mul(tf.reduce_sum(tf.square(x - mean), axes), divisor)
        return mean, var

    with tf.variable_scope(scope):
        mean, var = mean_var(x, axes=[0,1,2])
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]), name='beta', trainable=affine)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]), name='gamma', trainable=affine)
        normed = tf.nn.batch_norm_with_global_normalization(x, mean, var,
            beta, gamma, 1e-3, True)
    return normed


def residual_block(x, n_in, n_out, subsample, scope='res_block'):
    with tf.variable_scope(scope):
        if subsample:
            y = conv2d(x, n_in, n_out, 3, 2, 'SAME', False, scope='conv_1')
            shortcut = conv2d(x, n_in, n_out, 3, 2, 'SAME', False, scope='shortcut')
        else:
            y = conv2d(x, n_in, n_out, 3, 1, 'SAME', False, scope='conv_1')
            shortcut = tf.identity(x, name='shortcut')
        y = batch_norm(y, n_out, scope='bn_1')
        y = tf.nn.relu(y, name='relu_1')
        y = conv2d(y, n_out, n_out, 3, 1, 'SAME', True, scope='conv_2')
        y = y + shortcut
        y = tf.nn.relu(y, name='relu_2')
    return y


def residual_group(x, n_in, n_out, n, first_subsample, scope='res_group'):
    with tf.variable_scope(scope):
        y = residual_block(x, n_in, n_out, first_subsample, scope='block_1')
        for i in xrange(n-1):
            y = residual_block(y, n_out, n_out, False, scope='block_%d' % (i+2))
    return y


def residual_net(x, n, n_classes, scope='res_net'):
    with tf.variable_scope(scope):
        y = conv2d(x, 3, 16, 3, 1, 'SAME', False, scope='conv_init')
        y = batch_norm(y, 16, scope='bn_init')
        y = tf.nn.relu(y, name='relu_init')
        y = residual_group(y, 16, 16, n, False, scope='group_1')
        y = residual_group(y, 16, 32, n, True, scope='group_2')
        y = residual_group(y, 32, 64, n, True, scope='group_3')
        y = conv2d(y, 64, n_classes, 1, 1, 'SAME', True, scope='conv_last')
        y = tf.nn.avg_pool(y, [1,8,8,1], [1,1,1,1], 'VALID', name='avg_pool')
        y = tf.squeeze(y, squeeze_dims=[1,2])
    return y


def loss(logits, labels, scope='loss'):
    with tf.variable_scope(scope):
        targets = one_hot_embedding(labels, 10)
        entropy_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits, targets),
            name='entropy_loss')
        tf.add_to_collection('losses', entropy_loss)
        total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
    return total_loss


def accuracy(logits, gt_label, scope='accuracy'):
    with tf.variable_scope(scope):
        pred_label = tf.argmax(logits, 1)
        acc = 1.0 - tf.nn.zero_fraction(
            tf.cast(tf.equal(pred_label, gt_label), tf.int32))
    return acc


def train_op(train_loss, global_step, learning_rate):
    tf.scalar_summary('learning_rate', learning_rate)
    optim = tf.train.MomentumOptimizer(learning_rate, 0.9)
    params = tf.trainable_variables()
    gradients = tf.gradients(train_loss, params)
    updates = optim.apply_gradients(
        zip(gradients, params), global_step=global_step)
    with tf.control_dependencies([updates]):
        train_op = tf.no_op(name='train')
    return train_op


def cifar10_input_stream(records_path):
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer([records_path], None)
    _, record_value = reader.read(filename_queue)
    features = tf.parse_single_example(
        record_value,
        dense_keys=['image_raw', 'label'],
        dense_types=[tf.string, tf.int64])
    image = tf.decode_raw(features['image_raw'], tf.float32)
    image = tf.reshape(image, [32,32,3])
    label = tf.cast(features['label'], tf.int64)
    return image, label


def random_distort_image(image):
    distorted_image = image
    distorted_image = tf.image.pad_to_bounding_box(image, 4, 4, 40, 40) # pad 4 pixels to each side
    distorted_image = tf.image.random_crop(distorted_image, [32, 32])
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    # distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
    # distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
    # distorted_image = tf.image.per_image_whitening(distorted_image)
    return distorted_image


def make_train_batch(train_records_path, batch_size):
    with tf.variable_scope('train_batch'):
        with tf.device('/cpu:0'):
            train_image, train_label = cifar10_input_stream(train_records_path)
            train_image = random_distort_image(train_image)
            train_image_batch, train_label_batch = tf.train.shuffle_batch(
                [train_image, train_label], batch_size=batch_size, num_threads=4,
                capacity=50000,
                min_after_dequeue=1000)
    return train_image_batch, train_label_batch


def make_validation_batch(test_records_path, batch_size):
    with tf.variable_scope('evaluate_batch'):
        with tf.device('/cpu:0'):
            test_image, test_label = cifar10_input_stream(test_records_path)
            test_image_batch, test_label_batch = tf.train.batch(
                [test_image, test_label], batch_size=batch_size, num_threads=1,
                capacity=10000)
    return test_image_batch, test_label_batch


# class CnnClassifierModel(object):
#     def __init__(self, n_classes):
#         self.global_step = tf.Variable(0, trainable=False)
#         self.step_count = 0 # the value of global_step

#         with tf.variable_scope('inputs'):
#             self.ph_image = tf.placeholder(tf.float32)
#             self.ph_label = tf.placeholder(tf.int64)

#         self.learning_rate = tf.app.flags.FLAGS.learning_rate
#         log_dir = tf.app.flags.FLAGS.log_dir

#         n = 3
#         self.logits = residual_net(self.ph_image, n, n_classes)

#         with tf.variable_scope('loss'):
#             targets = one_hot_embedding(self.ph_label, n_classes)
#             entropy_loss = tf.reduce_mean(
#                 tf.nn.softmax_cross_entropy_with_logits(self.logits, targets),
#                 name='entropy_loss')
#             tf.add_to_collection('losses', entropy_loss)
#             self.loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
#         for var in tf.get_collection('losses'):
#             tf.scalar_summary(var.op.name, var)

#         with tf.variable_scope('accuracy'):
#             pred_label = tf.argmax(self.logits, 1)
#             accuracy = 1.0 - tf.nn.zero_fraction(
#                 tf.cast(tf.equal(pred_label, self.ph_label), tf.int32))
#             tf.scalar_summary('accuracy_train', accuracy)

#         with tf.variable_scope('gradients'):
#             params = tf.trainable_variables()
#             gradients = tf.gradients(self.loss, params)

#         with tf.variable_scope('optimizer'):
#             optim = tf.train.MomentumOptimizer(self.learning_rate, 0.9)
#             self.updates = optim.apply_gradients(
#                 zip(gradients, params), global_step=self.global_step)

#         # summaries for histograms for all trainable variables
#         for var in tf.trainable_variables():
#             tf.histogram_summary(var.op.name, var)

#         # summary writer
#         self.summary_writer = tf.train.SummaryWriter(log_dir)
#         self.merged_summary_op = tf.merge_all_summaries()

#         # saver
#         self.saver = tf.train.Saver(tf.all_variables())

#     def add_graph_summary(self, graph_def):
#         self.summary_writer.add_graph(graph_def)

#     def step(self, session, images, labels, do_summary=False):
#         input_feed = {}
#         input_feed[self.ph_image.name] = images
#         input_feed[self.ph_label.name] = labels
#         fetches = [
#             self.loss,
#             self.updates
#         ]
#         if do_summary:
#             fetches += [self.merged_summary_op]

#         session_outputs = session.run(fetches, input_feed)
#         loss_value = session_outputs[0]
#         self.step_count = int(self.global_step.eval(session))

#         if do_summary:
#             summary_str = session_outputs[2]
#             self.summary_writer.add_summary(summary_str, self.step_count)
#             self.summary_writer.flush()

#         return loss_value

#     def save(self, session, save_dir):
#         """
#         Save model to checkpoint.
#         Args:
#             session
#             save_dir: save checkpoint directory
#         """
#         self.saver.save(session, save_dir, global_step=self.step_count)


#     def load(self, session, load_dir):
#         """
#         Load model from checkpoint.
#         Args:
#             session
#             load_dir: load checkpoint directory
#         """
#         checkpoint = tf.train.get_checkpoint_state(load_dir)
#         self.saver.restore(session, checkpoint.model_checkpoint_path)
