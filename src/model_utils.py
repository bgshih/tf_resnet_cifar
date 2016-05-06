import tensorflow as tf

def shape_probe(tensor):
  return tf.Print(tensor, [tf.shape(tensor)], message='Shape=', summarize=10)

def min_max_probe(tensor):
  return tf.Print(tensor, [tf.reduce_min(tensor), tf.reduce_max(tensor)], message='Min, max=', summarize=10)

def conv_map_montage(conv_maps):
  """
  Montage of convolutional feature maps.

  Args:
    conv_maps: 4D tensor [B x H x W x C]
    maxWidth: maximum output width
    maxHeight: maximum output height
  Return:
    montage: [B x H' x W']
  """
  raise NotImplementedError
  # shape = tf.shape(conv_maps)
  # B, H, W, C = shape[0], shape[1], shape[2], shape[3]
  # maps = tf.transpose(conv_maps, [0,3,1,2])
  # tf.gather(maps, )

def activation_summary(x):
  tensor_name = x.op.name
  tf.histogram_summary('activations/' + tensor_name, x)
  # tf.scalar_summary(tensor_name + '/max', tf.reduce_max(x))
  # tf.scalar_summary(tensor_name + '/min', tf.reduce_min(x))
  # tf.scalar_summary(tensor_name + '/mean', tf.reduce_mean(x))
  tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def histogram_summary_for_all_variables():
  for var in tf.trainable_variables():
    tf.histogram_summary(var.op.name, var)

def add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.
  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.
  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summmary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.scalar_summary(l.op.name +' (raw)', l)
    tf.scalar_summary(l.op.name, loss_averages.average(l))

  return loss_averages_op
