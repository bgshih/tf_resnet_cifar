from __future__ import division

import sys
import os
import time
import math
import ipdb
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.python import control_flow_ops
import joblib

import model_resnet as m


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('load_dir', '', '')
tf.app.flags.DEFINE_integer('residual_net_n', 7, '')
tf.app.flags.DEFINE_string('train_tf_path', '../data/cifar10/train.tf', '')
tf.app.flags.DEFINE_string('val_tf_path', '../data/cifar10/test.tf', '')
tf.app.flags.DEFINE_integer('train_batch_size', 128, '')
tf.app.flags.DEFINE_integer('val_batch_size', 100, '')
tf.app.flags.DEFINE_float('weight_decay', 1e-4, 'Weight decay')
tf.app.flags.DEFINE_integer('summary_interval', 100, 'Interval for summary.')
tf.app.flags.DEFINE_integer('val_interval', 1000, 'Interval for evaluation.')
tf.app.flags.DEFINE_integer(
    'max_steps', 64000, 'Maximum number of iterations.')
tf.app.flags.DEFINE_string(
    'log_dir', '../logs_cifar10/log_%s' % time.strftime("%Y%m%d_%H%M%S"), '')
tf.app.flags.DEFINE_integer('save_interval', 5000, '')
tf.app.flags.DEFINE_string('restore_path', '/home/jrmei/research/tf_resnet_cifar/logs_cifar10/log_20160124_202605/checkpoint-499', 'the checkpoint to be restored')


def train_and_val():
    with tf.Graph().as_default():
        # train/test phase indicator
        phase_train = tf.placeholder(tf.bool, name='phase_train')

        # learning rate is manually set
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        # global step
        global_step = tf.Variable(0, trainable=False, name='global_step')

        # train/test inputs
        train_image_batch, train_label_batch = m.make_train_batch(
            FLAGS.train_tf_path, FLAGS.train_batch_size)
        val_image_batch, val_label_batch = m.make_validation_batch(
            FLAGS.val_tf_path, FLAGS.val_batch_size)
        image_batch, label_batch = control_flow_ops.cond(phase_train,
                                                         lambda: (
                                                             train_image_batch, train_label_batch),
                                                         lambda: (val_image_batch, val_label_batch))

        # model outputs
        logits = m.residual_net(
            image_batch, FLAGS.residual_net_n, 10, phase_train)

        # total loss
        m.loss(logits, label_batch)
        loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
        m.summary_losses()
        accuracy = m.accuracy(logits, label_batch)
        tf.scalar_summary('train_loss', loss)
        tf.scalar_summary('train_accuracy', accuracy)

        # saver
        saver = tf.train.Saver(tf.all_variables())

        # start session
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))

        # summary
        for var in tf.trainable_variables():
            tf.histogram_summary('params/' + var.op.name, var)

        init_op = tf.initialize_all_variables()
        if FLAGS.restore_path is None:
            # initialization
            print('Initializing...')
            sess.run(init_op, {phase_train.name: True})
        else:
            # restore from previous checkpoint
            sess.run(init_op, {phase_train.name: True})
            print('Restore variable from %s' % FLAGS.restore_path)
            saver.restore(sess, FLAGS.restore_path)

        # train loop
        tf.train.start_queue_runners(sess=sess)

        n_samples = 10000
        batch_size = FLAGS.val_batch_size
        n_iter = int(np.floor(n_samples / batch_size))
        accuracies = []
        losses = []
        for step in xrange(n_iter):
            fetches = [loss, accuracy]
            val_loss, val_acc = sess.run(
                fetches, {phase_train.name: False})
            losses.append(val_loss)
            accuracies.append(val_acc)
            print('[%s] Iteration %d, val loss = %f, val accuracy = %f' %
                  (datetime.now(), step, val_loss, val_acc))

        val_acc = np.mean(accuracies)
        val_loss = np.mean(losses)

        print('val losses is %f, accuracy is %f' % (val_loss, val_acc))


if __name__ == '__main__':
    train_and_val()
