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
tf.app.flags.DEFINE_integer('residual_net_n', 5, '')
tf.app.flags.DEFINE_string(
    'train_tf_path', '../data/cifar10/train.tf', '')
tf.app.flags.DEFINE_string(
    'val_tf_path', '../data/cifar10/test.tf', '')
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
#  tf.app.flags.DEFINE_integer('num_gpu', 2, '')
tf.app.flags.DEFINE_integer('num_gpu', 1, '')


def train_and_eval():
    with tf.Graph().as_default():
        # common part on cpu
        with tf.device('/cpu:0'):
            # train/test phase indicator
            phase_train = tf.placeholder(tf.bool, name='phase_train')

            # learning rate is manually set
            learning_rate = tf.placeholder(tf.float32, name='learning_rate')

            # global step
            global_step = tf.Variable(0, trainable=False, name='global_step')

            # optimizer
            learning_rate_weights = learning_rate
            learning_rate_biases = 2.0 * learning_rate  # double learning rate for biases
            optim_weights = tf.train.MomentumOptimizer(
                learning_rate_weights, 0.9)
            optim_biases = tf.train.MomentumOptimizer(
                learning_rate_biases, 0.9)

        gpu_grads = []
        # per gpu
        for i in xrange(FLAGS.num_gpu):
            print('Initialize the {0}th gpu'.format(i))
            with tf.device('/gpu:{0}'.format(i)):
                with tf.name_scope('gpu_{0}'.format(i)):
                    if i > 0:
                        m.add_to_collection = False

                    loss, accuracy, logits = loss_and_accuracy_per_gpu(
                        phase_train)

                    # Reuse variables
                    tf.get_variable_scope().reuse_variables()

                    weights, biases = tf.get_collection(
                        'weights'), tf.get_collection('biases')
                    #  print('weights len:{0}'.format(len(weights)))
                    #  print("len trainalbe variable:{0}".format(
                        #  len(tf.trainable_variables())))
                    assert(len(weights) + len(biases) == len(tf.trainable_variables()))

                    params = weights + biases
                    gradients = tf.gradients(loss, params, name='gradients')
                    #  gradient_weights = gradients[:len(weights)]
                    #  gradient_biases = gradients[len(weights):]

                    #  gpu_grads.append([gradient_weights, gradient_biases])
                    gpu_grads.append(gradients)

        with tf.device('/cpu:0'):
            # set up train_op
            weights, biases = tf.get_collection(
                'weights'), tf.get_collection('biases')
            averaged_grads = average_gradients(gpu_grads)
            weights_grads = averaged_grads[:len(weights)]
            biases_grads = averaged_grads[len(weights):]
            apply_weights_op = optim_weights.apply_gradients(
                zip(weights_grads, weights), global_step=global_step)
            apply_biases_op = optim_biases.apply_gradients(
                zip(biases_grads, biases), global_step=global_step)
            train_op = tf.group(apply_weights_op, apply_biases_op)

            # saver
            saver = tf.train.Saver(tf.all_variables())

            # start session
            sess = tf.Session(config=tf.ConfigProto(
                log_device_placement=False))

            # summary
            summary_op = tf.merge_all_summaries()
            summary_writer = tf.train.SummaryWriter(
                FLAGS.log_dir, graph_def=sess.graph_def)
            for var in tf.trainable_variables():
                tf.histogram_summary('params/' + var.op.name, var)

            # initialization (TODO: or load)
            init_op = tf.initialize_all_variables()

        print('Initializing...')
        sess.run(init_op, {phase_train.name: True})

        # train loop
        tf.train.start_queue_runners(sess=sess)
        curr_lr = 0.0
        lr_scale = 1.0
        for step in xrange(FLAGS.max_steps):
            # set learning rate manually
            if step <= 32000:
                _lr = lr_scale * 1e-1
            elif step <= 48000:
                _lr = lr_scale * 1e-2
            else:
                _lr = lr_scale * 1e-3
            if curr_lr != _lr:
                curr_lr = _lr
                print('Learning rate set to %f' % curr_lr)

            fetches = [train_op, loss]
            if step % FLAGS.summary_interval == 0:
                fetches += [accuracy, summary_op]
            sess_outputs = sess.run(
                fetches, {phase_train.name: True, learning_rate.name: curr_lr})

            print(sess_outputs)
            if step % FLAGS.summary_interval == 0:
                train_loss_value, train_acc_value, summary_str = sess_outputs[
                    1:]
                print('[%s] Iteration %d, train loss = %f, train accuracy = %f' %
                      (datetime.now(), step, train_loss_value, train_acc_value))
                summary_writer.add_summary(summary_str, step)

            if step > 0 and step % FLAGS.val_interval == 0:
                print('Evaluating...')
                n_val_samples = 10000
                val_batch_size = FLAGS.val_batch_size
                n_val_batch = int(n_val_samples / val_batch_size)
                val_losses = []
                val_accuracies = []
                for i in xrange(n_val_batch):
                    fetches = [accuracy, loss]
                    session_outputs = sess.run(
                        fetches, {phase_train.name: False})
                    val_accuracies.append(session_outputs[0])
                    val_losses.append(session_outputs[1])
                val_accuracy = float(np.mean(np.asarray(val_accuracies)))
                val_loss = float(np.mean(np.asarray(val_losses)))
                print('Test accuracy = %f' % val_accuracy)
                val_summary = tf.Summary()
                val_summary.value.add(tag='val_accuracy',
                                      simple_value=val_accuracy)
                val_summary.value.add(tag='val_loss', simple_value=val_loss)
                summary_writer.add_summary(val_summary, step)

            if step > 0 and step % FLAGS.save_interval == 0:
                checkpoint_path = os.path.join(FLAGS.log_dir, 'checkpoint')
                saver.save(sess, checkpoint_path, global_step=step)
                print('Checkpoint saved at %s' % checkpoint_path)


def average_gradients(gpu_grads):
    if len(gpu_grads) is 1:
        return gpu_grads[0]

    expand_grads = [[] for i in xrange(len(gpu_grads[0]))]
    average_grads = []
    for one_gpu_grad in gpu_grads:
        for i in xrange(len(one_gpu_grad)):
            expand_grads[i].append(tf.expand_dims(one_gpu_grad[i], 0))
    #  y = tf.concat(0, expand_grads)
    for x in xrange(len(expand_grads)):
        y = tf.concat(0, expand_grads[i])
        average_grad = tf.reduce_mean(y, 0)
        average_grads.append(average_grad)
    return average_grads


def loss_and_accuracy_per_gpu(phase_train):
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
    loss = m.loss(logits, label_batch)
    accuracy = m.accuracy(logits, label_batch)
    tf.scalar_summary('train_loss', loss)
    tf.scalar_summary('train_accuracy', accuracy)

    return loss, accuracy, logits

if __name__ == '__main__':
    pass
    train_and_eval()
