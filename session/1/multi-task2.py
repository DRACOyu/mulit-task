#coding=utf-8


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf 
import numpy as np 
import pdb
import os
from datetime import datetime
from create_tf_record import *
import tensorflow.contrib.slim as slim

from inception_v3 import *

from tensorflow.contrib import layers as layers_lib
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib import autograph

from utils import preprocessing

_WEIGHT_DECAY = 5e-4
_NUM_CLASSES = 5
_MIN_SCALE = 0.5
_MAX_SCALE = 2.0
_IGNORE_LABEL = 255

labels_nums = 1041  # 类别个数
labels_nums_seg = 5  # 类别个数
batch_size = 32  #
resize_height = 512  # 指定存储图片高度
resize_width = 170  # 指定存储图片宽度
depths = 3
data_shape = [batch_size, resize_height, resize_width, depths]


os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"

# 定义input_images为图片数据
input_images = tf.placeholder(dtype=tf.float32, shape=[None, resize_height, resize_width, depths], name='input')
# 定义input_labels为labels数据
# input_labels = tf.placeholder(dtype=tf.int32, shape=[None], name='label')
input_labels = tf.placeholder(dtype=tf.int32, shape=[None, labels_nums], name='label')
input_labels_seg = tf.placeholder(dtype=tf.int32, shape=[None,resize_height, resize_width,1], name='label_seg')
is_training = tf.placeholder(tf.bool, name='is_training')
reidseg = tf.placeholder(tf.bool, name='reidseg')

multi = tf.placeholder(dtype=tf.int32, name='label')

def get_batch_images_seg(images,labels,batch_size,labels_nums,one_hot=False,shuffle=False,num_threads=1):

    min_after_dequeue = 200
    capacity = min_after_dequeue + 3 * batch_size  # 保证capacity必须大于min_after_dequeue参数值
    if shuffle:
        images_batch, labels_batch = tf.train.shuffle_batch([images,labels],
                                                                    batch_size=batch_size,
                                                                    capacity=capacity,
                                                                    min_after_dequeue=min_after_dequeue,
                                                                    num_threads=num_threads)
    else:
        images_batch, labels_batch = tf.train.batch([images,labels],
                                                        batch_size=batch_size,
                                                        capacity=capacity,
                                                        num_threads=num_threads)
    return images_batch,labels_batch


def read_records_seg(filename,resize_height, resize_width,type=None):

    # 创建文件队列,不限读取的数量
    filename_queue = tf.train.string_input_producer([filename])
    # create a reader from file queue
    reader = tf.TFRecordReader()
    # reader从文件队列中读入一个序列化的样本
    _, serialized_example = reader.read(filename_queue)
    # 解析符号化的样本
    keys_to_features = {
        'image/height':
            tf.FixedLenFeature((), tf.int64),
        'image/width':
            tf.FixedLenFeature((), tf.int64),
        'image/encoded':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format':
            tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'label/encoded':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'label/format':
            tf.FixedLenFeature((), tf.string, default_value='png'),
    }

    features = tf.parse_single_example(serialized_example, keys_to_features)

    image = tf.image.decode_image(tf.reshape(features['image/encoded'], shape=[]), 3)
    image = tf.to_float(tf.image.convert_image_dtype(image, dtype=tf.uint8))
    image.set_shape([None, None, 3])

    label = tf.image.decode_image(tf.reshape(features['label/encoded'], shape=[]), 1)
    label = tf.to_int32(tf.image.convert_image_dtype(label, dtype=tf.uint8))
    label.set_shape([None, None, 1])

    image ,label = preprocess_image(image, label ,resize_height, resize_width,True)

    return image,label

def preprocess_image(image, label,resize_height, resize_width,is_training):
  """Preprocess a single image of layout [height, width, depth]."""
  if is_training:
    # Randomly scale the image and label.
    image, label = preprocessing.random_rescale_image_and_label(
        image, label, _MIN_SCALE, _MAX_SCALE)

    # Randomly crop or pad a [_HEIGHT, _WIDTH] section of the image and label.
    image, label = preprocessing.random_crop_or_pad_image_and_label(
        image, label, resize_height, resize_width, _IGNORE_LABEL)

    # Randomly flip the image and label horizontally.
    image, label = preprocessing.random_flip_left_right_image_and_label(
        image, label)

    image.set_shape([resize_height, resize_width, 3])
    label.set_shape([resize_height, resize_width, 1])
  image = preprocessing.mean_image_subtraction(image)

  return image, label

def atrous_spatial_pyramid_pooling(inputs, is_training, depth=256):
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


def _reduced_kernel_size_for_small_input(input_tensor, kernel_size):
  """Define kernel size which is automatically reduced for small input.

  If the shape of the input images is unknown at graph construction time this
  function assumes that the input images are is large enough.

  Args:
    input_tensor: input tensor of size [batch_size, height, width, channels].
    kernel_size: desired kernel size of length 2: [kernel_height, kernel_width]

  Returns:
    a tensor with the kernel size.

  TODO(jrru): Make this function work with unknown shapes. Theoretically, this
  can be done with the code below. Problems are two-fold: (1) If the shape was
  known, it will be lost. (2) inception.slim.ops._two_element_tuple cannot
  handle tensors that define the kernel size.
      shape = tf.shape(input_tensor)
      return = tf.stack([tf.minimum(shape[1], kernel_size[0]),
                         tf.minimum(shape[2], kernel_size[1])])

  """
  shape = input_tensor.get_shape().as_list()
  if shape[1] is None or shape[2] is None:
    kernel_size_out = kernel_size
  else:
    kernel_size_out = [shape[1],shape[2]]
  return kernel_size_out





def deeplab_v3_plus_generator(inputs, labels_nums,labels_nums_seg,is_training):

    with tf.contrib.slim.arg_scope(inception_v3_arg_scope()):
      net, end_points = inception_v3(inputs,
                                      num_classes=labels_nums,
                                      is_training=is_training,
                                      global_pool=False)


    inputs_size = tf.shape(inputs)[1:3]
    lip_net = end_points['dalated_inception7']
    with tf.variable_scope("decoder"):
      with tf.contrib.slim.arg_scope(inception_v3_arg_scope()):
        with arg_scope([layers.batch_norm], is_training=True):
          with tf.variable_scope('Logits'):
                kernel_size = _reduced_kernel_size_for_small_input(net, [8, 8])
                net = slim.avg_pool2d(net, kernel_size, padding='VALID',
                                      scope='AvgPool_1a_{}x{}'.format(*kernel_size))
                logits = slim.conv2d(net, labels_nums, [1, 1], activation_fn=None,
                                     normalizer_fn=None, scope='Conv2d_1c_1x1')
                logits_reid = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')

          with tf.variable_scope("low_level_features"):
            low_level_features = end_points['MaxPool_5a_3x3']
            low_level_features = layers_lib.conv2d(low_level_features, 48,
                                                   [1, 1], stride=1, scope='conv_1x1')
            low_level_features_size = tf.shape(low_level_features)[1:3]

          encoder_output = atrous_spatial_pyramid_pooling(lip_net, is_training)
          with tf.variable_scope("upsampling_logits"):
            net = tf.image.resize_bilinear(encoder_output, low_level_features_size, name='upsample_1')
            net = tf.concat([net, low_level_features], axis=3, name='concat')
            net = layers_lib.conv2d(net, 256, [3, 3], stride=1, scope='conv_3x3_1')
            net = layers_lib.conv2d(net, 256, [3, 3], stride=1, scope='conv_3x3_2')
            net = layers_lib.conv2d(net, labels_nums_seg, [1, 1], activation_fn=None, normalizer_fn=None, scope='conv_1x1')
            logits = tf.image.resize_bilinear(net, inputs_size, name='upsample_2')

    return logits,logits_reid

def net_evaluation(sess,loss,accuracy,val_images_batch,val_labels_batch,val_nums):
    val_max_steps = int(val_nums / batch_size)
    val_losses = []
    val_accs = []
    for _ in range(val_max_steps):
        val_x, val_y = sess.run([val_images_batch, val_labels_batch])
        # print('labels:',val_y)
        # val_loss = sess.run(loss, feed_dict={x: val_x, y: val_y, keep_prob: 1.0})
        # val_acc = sess.run(accuracy,feed_dict={x: val_x, y: val_y, keep_prob: 1.0})
        val_loss,val_acc = sess.run([loss,accuracy], feed_dict={input_images: val_x, input_labels: val_y, is_training: False})
        val_losses.append(val_loss)
        val_accs.append(val_acc)
    mean_loss = np.array(val_losses, dtype=np.float32).mean()
    mean_acc = np.array(val_accs, dtype=np.float32).mean()
    return mean_loss, mean_acc


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

    # If the number of valid entries is 0 (no classes) we return 0.
    result = tf.where(
        tf.greater(num_valid_entries, 0),
        tf.reduce_sum(iou, name=name) / num_valid_entries,
        0)
    return result

# @autograph.convert()
# def loss_acc(global_step_,input_labels,logits_reid,
#              input_labels_seg,logits,labels_nums_seg):
#
#     if multi == 0 :
#         # print("reid",global_step_)
#         # Specify the loss function: tf.losses定义的loss函数都会自动添加到loss函数,不需要add_loss()了
#         tf.losses.softmax_cross_entropy(onehot_labels=input_labels, logits=logits_reid)  # 添加交叉熵损失loss=1.6
#         # slim.losses.add_loss(my_loss)
#         loss = tf.losses.get_total_loss(add_regularization_losses=True)  # 添加正则化损失loss=2.2
#         # accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits_reid, 1), tf.argmax(input_labels, 1)), tf.float32))
#         train_mean_iou = 1.0
#         return loss,  train_mean_iou
#
#     else:        # print("seg",global_step_+1)
#         pred_classes = tf.expand_dims(tf.argmax(logits, axis=3, output_type = tf.int32), axis=3)#16,512,170,1
#         # train_var_list = [v for v in tf.trainable_variables()]
#         # loss_seg_L2 = _WEIGHT_DECAY * tf.add_n([tf.nn.l2_loss(v) for v in train_var_list])
#
#         labels_seg = tf.squeeze(input_labels_seg, axis=3)  # 16,512,170
#
#         logits_by_num_classes = tf.reshape(logits, [-1, labels_nums_seg])# ?,5
#         labels_flat = tf.reshape(labels_seg, [-1, ]) # 1392640
#
#         valid_indices = tf.to_int32(labels_flat <= labels_nums_seg - 1) # 1392640
#         valid_logits = tf.dynamic_partition(logits_by_num_classes, valid_indices, num_partitions=2)[1]
#         valid_labels = tf.dynamic_partition(labels_flat, valid_indices, num_partitions=2)[1]
#         # print("ok1")
#         preds_flat = tf.reshape(pred_classes, [-1, ])
#         valid_preds = tf.dynamic_partition(preds_flat, valid_indices, num_partitions=2)[1]
#         # print("ok1.1")
#         # accuracy = tf.metrics.accuracy(valid_labels, valid_preds)
#         mean_iou = tf.metrics.mean_iou(valid_labels, valid_preds, labels_nums_seg)
#         train_mean_iou = compute_mean_iou(mean_iou[1])
#         # print("ok2")
#         loss = tf.losses.sparse_softmax_cross_entropy(logits=valid_logits, labels=valid_labels)####+loss_seg_L2
#         # print("ok2.2")
#         # loss = tf.losses.get_total_loss(add_regularization_losses=True)
#         # print("ok3")
#         return loss,  train_mean_iou
#



def train(train_record_file,
          data_dir,
          train_log_step,
          train_param,
          val_record_file,
          val_log_step,
          labels_nums,
          labels_nums_seg,
          data_shape,
          snapshot,
          snapshot_prefix):

    [base_lr,max_steps]=train_param
    [batch_size,resize_height,resize_width,depths]=data_shape

    # 获得训练和测试的样本数
    train_nums=get_example_nums(train_record_file)
    val_nums=get_example_nums(val_record_file)
    print('train nums:%d,val nums:%d'%(train_nums,val_nums))

    # 从record中读取图片和labels数据
    train_images, train_labels = read_records(train_record_file, resize_height, resize_width, type='normalization')
    train_images_batch, train_labels_batch = get_batch_images(train_images, train_labels,
                                                              batch_size=batch_size, labels_nums=labels_nums,
                                                              one_hot=True, shuffle=True)
    val_images, val_labels = read_records(val_record_file, resize_height, resize_width, type='normalization')
    val_images_batch, val_labels_batch = get_batch_images(val_images, val_labels,
                                                          batch_size=batch_size, labels_nums=labels_nums,
                                                          one_hot=True, shuffle=False)


    train_seg_record_file = os.path.join(data_dir, 'LIP_train5.record')
    val_seg_record_file = os.path.join(data_dir,  'LIP_val5.record')
    train_images_seg, train_labels_seg = read_records_seg(train_seg_record_file, resize_height, resize_width, type='normalization')
    train_images_batch_seg, train_labels_batch_seg = get_batch_images_seg(train_images_seg, train_labels_seg,
                                                                          batch_size=batch_size, labels_nums=labels_nums_seg,
                                                                          shuffle=True)
    # val_images_seg, val_labels_seg = read_records_seg(val_seg_record_file, resize_height, resize_width, type='normalization')
    # val_images_batch_seg, val_labels_batch_seg = get_batch_images_seg(val_images_seg, val_labels_seg,
    #                                                       batch_size=batch_size, labels_nums=labels_nums_seg,
    #                                                       one_hot=True, shuffle=False)


    logits, logits_reid = deeplab_v3_plus_generator(input_images,labels_nums,labels_nums_seg,is_training)
    global_step_ = tf.Variable(tf.constant(0))

    # Specify the loss function: tf.losses定义的loss函数都会自动添加到loss函数,不需要add_loss()了
    loss = tf.losses.softmax_cross_entropy(onehot_labels=input_labels, logits=logits_reid)  # 添加交叉熵损失loss=1.6
    # slim.losses.add_loss(my_loss)
    # loss = tf.losses.get_total_loss(add_regularization_losses=True)  # 添加正则化损失loss=2.2

    pred_classes = tf.expand_dims(tf.argmax(logits, axis=3, output_type = tf.int32), axis=3)#16,512,170,1
    # train_var_list = [v for v in tf.trainable_variables()]
    # loss_seg_L2 = _WEIGHT_DECAY * tf.add_n([tf.nn.l2_loss(v) for v in train_var_list])

    labels_seg = tf.squeeze(input_labels_seg, axis=3)  # 16,512,170

    logits_by_num_classes = tf.reshape(logits, [-1, labels_nums_seg])# ?,5
    labels_flat = tf.reshape(labels_seg, [-1, ]) # 1392640

    valid_indices = tf.to_int32(labels_flat <= labels_nums_seg - 1) # 1392640
    valid_logits = tf.dynamic_partition(logits_by_num_classes, valid_indices, num_partitions=2)[1]
    valid_labels = tf.dynamic_partition(labels_flat, valid_indices, num_partitions=2)[1]
    # print("ok1")
    preds_flat = tf.reshape(pred_classes, [-1, ])
    valid_preds = tf.dynamic_partition(preds_flat, valid_indices, num_partitions=2)[1]
    # print("ok1.1")
    # accuracy = tf.metrics.accuracy(valid_labels, valid_preds)
    mean_iou = tf.metrics.mean_iou(valid_labels, valid_preds, labels_nums_seg)
    train_mean_iou = compute_mean_iou(mean_iou[1])
    # print("ok2")
    loss_seg = tf.losses.sparse_softmax_cross_entropy(logits=valid_logits, labels=valid_labels)####+loss_seg_L2
    # loss_seg = tf.losses.get_total_loss(add_regularization_losses=True)  # 添加正则化损失loss=2.2



    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits_reid, 1), tf.argmax(input_labels, 1)), tf.float32))

    # loss,  train_mean_iou =loss_acc( global_step_,input_labels, logits_reid,
    #                                           input_labels_seg, logits, labels_nums_seg)

    base_lr = tf.train.exponential_decay(base_lr, global_step_, 10000, 0.95)
    optimizer = tf.train.MomentumOptimizer(learning_rate=base_lr,momentum= 0.9)
    # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    # 使用`tensorflow`的控制流, 先执行更新算子, 再执行训练
    # # 定义优化算子
    # optimizer = tf.train.AdamOptimizer(1e-3)
    # #选择待优化的参数
    # output_vars = tf.get_collection(tf.GraphKyes.TRAINABLE_VARIABLES, scope='outpt')
    # train_step = optimizer.minimize(loss_score,var_list = output_vars)
    # Batch norm requires update ops to be added as a dependency to the train_op

    update_list2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,  scope= "decoder")
    update_list3 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,  scope=  'dalated_reduction7')
    update_list4 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,  scope= 'dalated_inception7')
    update_list = update_list2 + update_list3 + update_list4

    update_list_reid = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='Logits')
    # update_list_reid2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,  scope= "decoder")
    # update_list3 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,  scope=  'InceptionV3/dalated_reduction7')
    # update_list4 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,  scope= 'InceptionV3/dalated_inception7')
    # update_list_reid = update_list1 +update_list2 + update_list3 + update_list4

    # with tf.control_dependencies(update_ops):
        # train_op = optimizer.minimize(loss, global_step_, )#var_list=update_list_reid
        # train_op_seg = optimizer.minimize(loss_seg, global_step_, )#var_list=update_list
    train_op = slim.learning.create_train_op(total_loss=loss, optimizer=optimizer,update_ops = update_list_reid)
    train_op_seg = slim.learning.create_train_op(total_loss=loss_seg, optimizer=optimizer,update_ops = None)

    # Pretrained_model_dir = "/yzc/test/tensorflow-inceptionv3-reid-9/models/reid-9-imageft/model_ckpt-28w+.ckpt"
    Pretrained_model_dir = '/yzc/test/test_module/models/sp-reid-ft/512/best_models_66000_0.9350.ckpt'
    exclusions = ['InceptionV3/AuxLogits', "decoder",'InceptionV3/Logits',"Logits",
               'InceptionV3/dalated_reduction7', 'InceptionV3/dalated_inception7']

    inception_except_logits = slim.get_variables_to_restore(exclude=exclusions)


    saver = tf.train.Saver(inception_except_logits)
    max_acc = 0.0
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        saver.restore(sess, Pretrained_model_dir)
        for i in range(max_steps + 1):
            batch_input_images, batch_input_labels = sess.run([train_images_batch, train_labels_batch])
            batch_input_images_seg, batch_input_labels_seg = sess.run([train_images_batch_seg, train_labels_batch_seg])

            if i % 2 ==0:
                _, train_loss= sess.run([train_op, loss], feed_dict={input_images: batch_input_images,
                                                                         input_labels: batch_input_labels,
                                                                         input_labels_seg: batch_input_labels_seg,
                                                                         global_step_: i ,is_training: True,multi:0})
                # print("%s: Step [%d]  train Loss : %f" % (datetime.now(), i,train_loss))
                if i % train_log_step == 0:
                    train_acc = sess.run(accuracy, feed_dict={input_images: batch_input_images,
                                                              input_labels: batch_input_labels,
                                                              is_training: False})
                    print("%s: Step [%d]  train Loss : %f,training accuracy :  %g" % (
                        datetime.now(), i, train_loss, train_acc))
            else:
                _, train_loss_seg= sess.run([train_op_seg, loss_seg], feed_dict={input_images: batch_input_images_seg,
                                                                         input_labels: batch_input_labels,
                                                                         input_labels_seg: batch_input_labels_seg,
                                                                         global_step_: i,is_training: True,multi:1})
                # print("%s: Step [%d]  train_loss_seg : %f" % (datetime.now(), i, train_loss_seg))
                if (i-1) % train_log_step == 0:
                    train_acc = sess.run(train_mean_iou, feed_dict={input_images: batch_input_images_seg,
                                                                    input_labels: batch_input_labels,
                                                                    input_labels_seg: batch_input_labels_seg,
                                                                    global_step_: i,is_training: False,multi:1})
                    print("%s: Step [%d]  train_loss_seg : %f,training accuracy :  %g" % (
                        datetime.now(), i, train_loss_seg, train_acc))

            # # train测试(这里仅测试训练集的一个batch)global_step_: num_lr,
            # if i % train_log_step == 0:
            #     train_acc = sess.run(accuracy, feed_dict={input_images: batch_input_images,
            #                                               input_labels: batch_input_labels,
            #                                               is_training: False})
            #     print("%s: Step [%d]  train Loss : %f, training loss_seg :  %f,training accuracy :  %g" % (
            #     datetime.now(), i, train_loss,train_loss_seg, train_acc))

            # # val测试(测试全部val数据)
            # if i % val_log_step == 0:
            #     mean_loss, mean_acc = net_evaluation(sess, loss, accuracy, val_images_batch, val_labels_batch, val_nums)
            #     print("%s: Step [%d]  val Loss : %f, val accuracy :  %g" % (datetime.now(), i, mean_loss, mean_acc))
            # 模型保存:每迭代snapshot次或者最后一次保存模型

            if (i % snapshot == 0 and i > 0) or i == max_steps:
                print('-----save:{}-{}'.format(snapshot_prefix, i))
                saver.save(sess, snapshot_prefix, global_step=i)

            # # 保存val准确率最高的模型
            # # if mean_acc > max_acc and mean_acc > 0.93:
            # #     max_acc = mean_acc
            # #     path = os.path.dirname(snapshot_prefix)
            # #     best_models = os.path.join(path, 'best_models_{}_{:.4f}.ckpt'.format(i, max_acc))
            # #     print('------save:{}'.format(best_models))
            # #     saver.save(sess, best_models)

        coord.request_stop()
        coord.join(threads)
    # 循环迭代过程


if __name__ == '__main__':
    train_record_file='/root/data/MSMT17_V1/record/train-512-170.tfrecords'
    val_record_file='/root/data/MSMT17_V1/record/val-512-170.tfrecords'
    data_dir= '/yzc/test/inceptionv3-msmt17-multi-dilated/dataset/'

    train_log_step=100
    base_lr = 0.01  # 学习率
    max_steps = 200000  # 迭代次数
    train_param=[base_lr,max_steps]

    val_log_step=2
    snapshot=1000#保存文件间隔
    snapshot_prefix='/yzc/test/test_module/models-multi-1/model'
    train(train_record_file=train_record_file,
          data_dir=data_dir,
          train_log_step=train_log_step,
          train_param=train_param,
          val_record_file=val_record_file,
          val_log_step=val_log_step,
          labels_nums=labels_nums,
          labels_nums_seg=labels_nums_seg,
          data_shape=data_shape,
          snapshot=snapshot,
          snapshot_prefix=snapshot_prefix)
