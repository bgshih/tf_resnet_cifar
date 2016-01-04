import tensorflow as tf
import math


def weighted_kernel_conv(x, n_in, n_out, ksize, stride, scope=None):
    scope = scope or "wkcnn"
    with tf.variable_scope(scope):
        kernel = tf.Variable(
            tf.truncated_normal([ksize, ksize, n_in, n_out],
                stddev=math.sqrt(2/(ksize*ksize*n_out))))
        bias = tf.Variable(tf.zeros([n_out]))

        kernel_weights = tf.Variable(tf.ones([n_in, n_out]))
        norm_kernel_weights = tf.transpose(tf.nn.softmax(tf.transpose(kernel_weights)))
        wkernel = kernel * tf.reshape(norm_kernel_weights, [1, 1, n_in, n_out])

        conv = tf.nn.conv2d(x, wkernel, [1,stride,stride,1], padding='SAME')
    return conv


