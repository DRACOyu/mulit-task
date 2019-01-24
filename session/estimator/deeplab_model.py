"""DeepLab v3 models based on slim library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from inception_v3 import *

import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
# import slim.nets.inception_v3 as inception_v3

from tensorflow.contrib import layers as layers_lib
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers

from utils import preprocessing

_BATCH_NORM_DECAY = 0.9997
_WEIGHT_DECAY = 5e-4
num = 1041

def atrous_spatial_pyramid_pooling(inputs, batch_norm_decay, is_training, depth=256):
  """Atrous Spatial Pyramid Pooling.

  Args:
    inputs: A tensor of size [batch, height, width, channels].
    output_stride: The ResNet unit's stride. Determines the rates for atrous convolution.
      the rates are (6, 12, 18) when the stride is 16, and doubled when 8.
    batch_norm_decay: The moving average decay when estimating layer activation
      statistics in batch normalization.
    is_training: A boolean denoting whether the input is for training.
    depth: The depth of the ResNet unit output.

  Returns:
    The atrous spatial pyramid pooling output.
  """
  with tf.variable_scope("aspp"):
    # if output_stride not in [8, 16]:
    #   raise ValueError('output_stride must be either 8 or 16.')

    atrous_rates = [3, 4, 6]
    # if output_stride == 8:
    #   atrous_rates = [2*rate for rate in atrous_rates]

    with tf.contrib.slim.arg_scope(inception_v3_arg_scope()):
      with arg_scope([layers.batch_norm], is_training=is_training):
        inputs_size = tf.shape(inputs)[1:3]
        # (a) one 1x1 convolution and three 3x3 convolutions with rates = (6, 12, 18) when output stride = 16.
        # the rates are doubled when output stride = 8.
        conv_1x1 = layers_lib.conv2d(inputs, depth, [1, 1], stride=1, scope="conv_1x1")
        conv_3x3_1 = layers_lib.conv2d(inputs, depth, [3, 3], stride=1, rate=atrous_rates[0], scope='conv_3x3_1')
        conv_3x3_2 = layers_lib.conv2d(inputs, depth, [3, 3], stride=1, rate=atrous_rates[1], scope='conv_3x3_2')
        conv_3x3_3 = layers_lib.conv2d(inputs, depth, [3, 3], stride=1, rate=atrous_rates[2], scope='conv_3x3_3')

        # (b) the image-level features
        with tf.variable_scope("image_level_features"):
          # global average pooling
          image_level_features = tf.reduce_mean(inputs, [1, 2], name='global_average_pooling', keepdims=True)
          # 1x1 convolution with 256 filters( and batch normalization)
          image_level_features = layers_lib.conv2d(image_level_features, depth, [1, 1], stride=1, scope='conv_1x1')
          # bilinearly upsample features
          image_level_features = tf.image.resize_bilinear(image_level_features, inputs_size, name='upsample')

        net = tf.concat([conv_1x1, conv_3x3_1, conv_3x3_2, conv_3x3_3, image_level_features], axis=3, name='concat')
        net = layers_lib.conv2d(net, depth, [1, 1], stride=1, scope='conv_1x1_concat')

        return net


def deeplab_v3_plus_generator(num_classes,
                              pre_trained_model,
                              batch_norm_decay,
                              data_format='channels_last'):
  """Generator for DeepLab v3 plus models.

  Args:
    num_classes: The number of possible classes for image classification.
    output_stride: The ResNet unit's stride. Determines the rates for atrous convolution.
      the rates are (6, 12, 18) when the stride is 16, and doubled when 8.
    base_architecture: The architecture of base Resnet building block.
    pre_trained_model: The path to the directory that contains pre-trained models.
    batch_norm_decay: The moving average decay when estimating layer activation
      statistics in batch normalization.
    data_format: The input format ('channels_last', 'channels_first', or None).
      If set to None, the format is dependent on whether a GPU is available.
      Only 'channels_last' is supported currently.

  Returns:
    The model function that takes in `inputs` and `is_training` and
    returns the output tensor of the DeepLab v3 model.
  """


  if batch_norm_decay is None:
    batch_norm_decay = _BATCH_NORM_DECAY

  def model(inputs, is_training):
    # encoder
    num = 1041
    with tf.contrib.slim.arg_scope(inception_v3_arg_scope()):
      logitss, end_points = inception_v3(inputs,
                                      num_classes=num,
                                      is_training=is_training,
                                      global_pool=False)

    # logits_reid,logitssegseg = tf.split(end_points['Logits'], 2, 0)
    # print(logits_reid)
    # logits_reid = end_points['Logits'][:16]
    # logits_reid = end_points['Logits']
      logits_reid = logitss[:16]

    if is_training:
      exclude = ['InceptionV3/AuxLogits','InceptionV3/Logits',"aspp","decoder",
                 'InceptionV3/dalated_reduction7','InceptionV3/dalated_inception7']
      ##"Conv2d_1c_1x1/biases",
      # pre_trained_model = tf.train.latest_checkpoint(pre_trained_model)
      variables_to_restore = tf.contrib.slim.get_variables_to_restore(exclude=exclude)
      tf.train.init_from_checkpoint(pre_trained_model,
                                    {v.name.split(':')[0]: v for v in variables_to_restore})


    inputs_size = tf.shape(inputs)[1:3]
    # net = end_points[base_architecture + '/block4']
    # lip_net = end_points['Mixed_6e'][:8]

    ##add dilated
    lip_net = end_points['dalated_inception7'][16:]
    # lip_net = end_points["Mixed_6e"]


    with tf.variable_scope("decoder",reuse=tf.AUTO_REUSE):
      with tf.contrib.slim.arg_scope(inception_v3_arg_scope()):
        with arg_scope([layers.batch_norm], is_training=is_training):
          with tf.variable_scope("low_level_features"):
            low_level_features = end_points['MaxPool_5a_3x3'][16:]
            # low_level_features = end_points['MaxPool_5a_3x3']
            low_level_features = layers_lib.conv2d(low_level_features, 48,
                                                   [1, 1], stride=1, scope='conv_1x1')
            low_level_features_size = tf.shape(low_level_features)[1:3]


          encoder_output = atrous_spatial_pyramid_pooling(lip_net, batch_norm_decay, is_training)
          with tf.variable_scope("upsampling_logits"):
            net = tf.image.resize_bilinear(encoder_output, low_level_features_size, name='upsample_1')
            net = tf.concat([net, low_level_features], axis=3, name='concat')
            net = layers_lib.conv2d(net, 256, [3, 3], stride=1, scope='conv_3x3_1')
            net = layers_lib.conv2d(net, 256, [3, 3], stride=1, scope='conv_3x3_2')
            net = layers_lib.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='conv_1x1')
            logits = tf.image.resize_bilinear(net, inputs_size, name='upsample_2')

    return logits,logits_reid

  return model


def deeplabv3_plus_model_fn(features, labels, mode, params):
  """Model function for PASCAL VOC."""
  if isinstance(features, dict):
    features_seg = features['seg']
    features_reid = features['reid']
  if isinstance(labels, dict):
    labels_seg=labels["seg"]
    labels_reid =labels["reid"]

  # labels_reid ,error = tf.split(labels_reid, 2, 0)
  # print(labels_reid)
  # label_zero = tf.zeros([16,1041],tf.int32)
  # labels_reid = tf.concat([labels_reid,label_zero],axis=0)
  # print (labels_reid)

  # shape = features_reid.shape[0]
  images = tf.cast(
      tf.map_fn(preprocessing.mean_image_addition, features_seg),
      tf.uint8)
  images_reid = tf.cast(
      tf.map_fn(preprocessing.mean_image_addition, features_reid),
      tf.uint8)

  network = deeplab_v3_plus_generator(params['num_classes'],
                                      params['pre_trained_model'],
                                      params['batch_norm_decay'])

  # features_reid = features_reid.eval(session=sess)
  # features = np.concatenate(0,[features_reid,features_reid],axis=0)
  # features = tf.convert_to_tensor(features)
  # features = tf.concat([features_reid,features_reid],axis=0)
  features = tf.concat([features_reid,features_seg],axis=0)
  # features = features_reid

  # logits,logits_reid_= network(features_seg, mode == tf.estimator.ModeKeys.TRAIN)
  # logits_,logits_reid= network(features_reid, mode == tf.estimator.ModeKeys.TRAIN)
  logits,logits_reid= network(features, mode == tf.estimator.ModeKeys.TRAIN)

  # logits_,logits_reid = network(features_reid, mode == tf.estimator.ModeKeys.TRAIN)#tf.estimator.ModeKeys.TRAIN

  pred_classes = tf.expand_dims(tf.argmax(logits, axis=3, output_type=tf.int32), axis=3)

  pred_decoded_labels = tf.py_func(preprocessing.decode_labels,
                                   [pred_classes, params['batch_size'], params['num_classes']],
                                   tf.uint8)

  predictions = {
      'classes': pred_classes,
      'probabilities': tf.nn.softmax(logits_reid, name='softmax_tensor'),
      'decoded_labels': pred_decoded_labels
  }

  gt_decoded_labels = tf.py_func(preprocessing.decode_labels,
                                 [labels_seg, params['batch_size'], params['num_classes']], tf.uint8)

  labels_seg = tf.squeeze(labels_seg, axis=3)  # reduce the channel dimension.


  logits_by_num_classes = tf.reshape(logits, [-1, params['num_classes']])
  labels_flat = tf.reshape(labels_seg, [-1, ])

  valid_indices = tf.to_int32(labels_flat <= params['num_classes'] - 1)
  valid_logits = tf.dynamic_partition(logits_by_num_classes, valid_indices, num_partitions=2)[1]
  valid_labels = tf.dynamic_partition(labels_flat, valid_indices, num_partitions=2)[1]

  preds_flat = tf.reshape(pred_classes, [-1, ])
  valid_preds = tf.dynamic_partition(preds_flat, valid_indices, num_partitions=2)[1]
  confusion_matrix = tf.confusion_matrix(valid_labels, valid_preds, num_classes=params['num_classes'])

  predictions['valid_preds'] = valid_preds
  predictions['valid_labels'] = valid_labels
  predictions['confusion_matrix'] = confusion_matrix

  global_step = tf.train.get_or_create_global_step()
  global_step_ = tf.cast(global_step, tf.int32) - params['initial_global_step']

  # if labels_reid.shape[1]==1041:
  if global_step_ % 200 == 0:
      # Specify the loss function: tf.losses定义的loss函数都会自动添加到loss函数,不需要add_loss()了
      loss_reid = tf.losses.softmax_cross_entropy(onehot_labels=labels_reid, logits=logits_reid) # 添加交叉熵损失loss=1.6
      cross_entropy = tf.losses.sparse_softmax_cross_entropy(logits=valid_logits, labels=valid_labels)* 0
  else:
      loss_reid = tf.losses.softmax_cross_entropy(onehot_labels=labels_reid, logits=logits_reid)* 0 # 添加交叉熵损失loss=1.6
      cross_entropy = tf.losses.sparse_softmax_cross_entropy(logits=valid_logits, labels=valid_labels)
  # slim.losses.add_loss(my_loss)
  loss = tf.losses.get_total_loss(add_regularization_losses=True)  # 添加正则化损失loss=2.2
  # accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(reid_logits, 1), tf.argmax(input_labels, 1)), tf.float32))

  # Create a tensor named cross_entropy for logging purposes.
  tf.identity(cross_entropy, name='cross_entropy')
  tf.summary.scalar('cross_entropy', cross_entropy)

  # Create a tensor named cross_entropy for logging purposes.
  tf.identity(loss_reid, name='loss_reid')
  tf.summary.scalar('loss_reid', loss_reid)

  tf.identity(loss, name='loss')
  tf.summary.scalar('loss', loss)


  if not params['freeze_batch_norm']:
    train_var_list = [v for v in tf.trainable_variables()]
  else:
    train_var_list = [v for v in tf.trainable_variables()
                      if 'beta' not in v.name and 'gamma' not in v.name]


  # Add weight decay to the loss.
  # with tf.variable_scope("total_loss"):
    # loss =loss_reid+ cross_entropy + params.get('weight_decay', _WEIGHT_DECAY) * tf.add_n(
    #     [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in train_var_list])
    # loss = loss_reid + cross_entropy\
           # +params.get('weight_decay', _WEIGHT_DECAY) * tf.add_n([tf.nn.l2_loss(v) for v in train_var_list])
    # loss = loss_reid
    # loss = tf.losses.get_total_loss()  # obtain the regularization losses as well

  if mode == tf.estimator.ModeKeys.TRAIN:

    tf.summary.image('images',
                     tf.concat(axis=2, values=[images, gt_decoded_labels, pred_decoded_labels]),
                     max_outputs=10)  # Concatenate row-wise.

    tf.summary.image('images',images_reid,max_outputs=10)  # Concatenate row-wise.

    # global_step = tf.train.get_or_create_global_step()

    # global_step = tf.train.get_or_create_global_step()

    base_lr=0.01
    global_step_ = tf.cast(global_step, tf.int32)-params['initial_global_step']
    # if (global_step_ + 1) % int(10) == 0:
        # decay = ((1.0 - float(global_step_) / max_steps) ** 0.9)
        # decay = 0.9
        # base_lr = base_lr * decay
    # base_lr = tf.train.exponential_decay(base_lr, tf.cast(global_step, tf.int32) - params['initial_global_step'], 10000, 0.95)
    # Create a tensor named learning_rate for logging purposes
    learning_rate = tf.train.exponential_decay(base_lr, global_step_, 10000, 0.95)

    tf.identity(learning_rate, name='learning_rate')
    tf.summary.scalar('learning_rate', learning_rate)

    tf.identity(global_step_, name='global_step_')
    tf.summary.scalar('global_step_', global_step_)

    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate,
        momentum=params['momentum'])

    # # 定义优化算子
    # optimizer = tf.train.AdamOptimizer(1e-3)
    # #选择待优化的参数
    # output_vars = tf.get_collection(tf.GraphKyes.TRAINABLE_VARIABLES, scope='outpt')
    # train_step = optimizer.minimize(loss_score,var_list = output_vars)
    # Batch norm requires update ops to be added as a dependency to the train_op

    # update_list1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="aspp")
    # update_list2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,  scope= "decoder")
    # update_list3 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,  scope=  'InceptionV3/dalated_reduction7')
    # update_list4 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,  scope= 'InceptionV3/dalated_inception7')
    # update_list = update_list1 +update_list2 + update_list3 + update_list4
    # print(update_list)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss, global_step)
      # train_op = optimizer.minimize(loss, global_step, var_list=update_list)
      # train_op = slim.learning.create_train_op(total_loss=loss, optimizer=optimizer)
  else:
    train_op = None

  accuracy = tf.metrics.accuracy(
      valid_labels, valid_preds)
  mean_iou = tf.metrics.mean_iou(valid_labels, valid_preds, params['num_classes'])
  labels_reid =tf.cast(labels_reid,tf.int64)
  acc3= tf.metrics.precision_at_k(labels_reid,logits_reid,num)
  labels_reid =tf.cast(labels_reid,tf.float32)
  acc2 = tf.metrics.mean_squared_error(labels_reid,logits_reid)
  metrics = {'px_accuracy': accuracy, 'mean_iou': mean_iou,"acc2":acc2,"acc3":acc3}##
  #
  # # Create a tensor named train_accuracy for logging purposes
  tf.identity(accuracy[1], name='train_px_accuracy')
  tf.summary.scalar('train_px_accuracy', accuracy[1])

  accuracy_reid = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits_reid, 1), tf.argmax(labels_reid, 1)), tf.float32))

  tf.identity(accuracy_reid, name='accuracy_reid')
  tf.summary.scalar('accuracy_reid', accuracy_reid)



  def compute_mean_iou(total_cm, name='mean_iou'):
    """Compute the mean intersection-over-union via the confusion matrix."""
    sum_over_row = tf.to_float(tf.reduce_sum(total_cm, 0))
    sum_over_col = tf.to_float(tf.reduce_sum(total_cm, 1))
    cm_diag = tf.to_float(tf.diag_part(total_cm))
    denominator = sum_over_row + sum_over_col - cm_diag

    # The mean is only computed over classes that appear in the
    # label or prediction tensor. If the denominator is 0, we need to
    # ignore the class.
    num_valid_entries = tf.reduce_sum(tf.cast(
        tf.not_equal(denominator, 0), dtype=tf.float32))

    # If the value of the denominator is 0, set it to 1 to avoid
    # zero division.
    denominator = tf.where(
        tf.greater(denominator, 0),
        denominator,
        tf.ones_like(denominator))
    iou = tf.div(cm_diag, denominator)

    for i in range(params['num_classes']):
      tf.identity(iou[i], name='train_iou_class{}'.format(i))
      tf.summary.scalar('train_iou_class{}'.format(i), iou[i])

    # If the number of valid entries is 0 (no classes) we return 0.
    result = tf.where(
        tf.greater(num_valid_entries, 0),
        tf.reduce_sum(iou, name=name) / num_valid_entries,
        0)
    return result

  train_mean_iou = compute_mean_iou(mean_iou[1])

  tf.identity(train_mean_iou, name='train_mean_iou')
  tf.summary.scalar('train_mean_iou', train_mean_iou)

  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=metrics)
