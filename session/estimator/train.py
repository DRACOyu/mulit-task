"""Train a DeepLab v3 plus model using tf.estimator API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf
import deeplab_model
from utils import preprocessing
from tensorflow.python import debug as tf_debug
from create_tf_record import *

import shutil



os.environ["CUDA_VISIBLE_DEVICES"] = "5"
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"



parser = argparse.ArgumentParser()

parser.add_argument('--model_dir', type=str, default='./model_0116_multi_testdata_1',
                    help='Base directory for the model.')

parser.add_argument('--clean_model_dir', action='store_true',
                    help='Whether to clean up the model directory if present.')

parser.add_argument('--train_epochs', type=int, default=45,
                    help='Number of training epochs: '
                         'For 30K iteration with batch size 6, train_epoch = 17.01 (= 30K * 6 / 10,582). '
                         'For 30K iteration with batch size 8, train_epoch = 22.68 (= 30K * 8 / 10,582). '
                         'For 30K iteration with batch size 10, train_epoch = 25.52 (= 30K * 10 / 10,582). '
                         'For 30K iteration with batch size 11, train_epoch = 31.19 (= 30K * 11 / 10,582). '
                         'For 30K iteration with batch size 15, train_epoch = 42.53 (= 30K * 15 / 10,582). '
                         'For 30K iteration with batch size 16, train_epoch = 45.36 (= 30K * 16 / 10,582).')

parser.add_argument('--epochs_per_eval', type=int, default=1,
                    help='The number of training epochs to run between evaluations.')

parser.add_argument('--tensorboard_images_max_outputs', type=int, default=6,
                    help='Max number of batch elements to generate for Tensorboard.')

parser.add_argument('--batch_size', type=int, default=16,
                    help='Number of examples per batch.')

parser.add_argument('--learning_rate_policy', type=str, default='poly',
                    choices=['poly', 'piecewise'],
                    help='Learning rate policy to optimize loss.')

parser.add_argument('--max_iter', type=int, default=30000,
                    help='Number of maximum iteration used for "poly" learning rate policy.')

parser.add_argument('--data_dir', type=str, default='/yzc/test/inceptionv3-msmt17-multi-dilated/dataset/',
                    help='Path to the directory containing the PASCAL VOC data tf record.')

parser.add_argument('--reid_data_dir', type=str, default='/root/data/MSMT17_V1/record/',
                    help='Path to the directory containing the PASCAL VOC data tf record.')

parser.add_argument('--pre_trained_model', type=str, default='/yzc/test/tensorflow-inceptionv3-reid-9/models/reid-9-imageft/model_ckpt-28w+.ckpt',
                    help='Path to the pre-trained model checkpoint.')

parser.add_argument('--freeze_batch_norm', action='store_true',
                    help='Freeze batch normalization parameters during the training.')

parser.add_argument('--initial_learning_rate', type=float, default=0.01,
                    help='Initial learning rate for the optimizer.')

parser.add_argument('--end_learning_rate', type=float, default=1e-6,
                    help='End learning rate for the optimizer.')

parser.add_argument('--initial_global_step', type=int, default=286003,
                    help='Initial global step for controlling learning rate when fine-tuning model.')

parser.add_argument('--weight_decay', type=float, default=2e-4,
                    help='The weight decay to use for regularizing the model.')

parser.add_argument('--debug', action='store_true',
                    help='Whether to use debugger to track down bad values during training.')

_NUM_CLASSES = 5
_HEIGHT = 512
_WIDTH = 170
_DEPTH = 3
_MIN_SCALE = 0.5
_MAX_SCALE = 2.0
_IGNORE_LABEL = 255

_MOMENTUM = 0.9


labels_nums=1041
_BATCH_NORM_DECAY = 0.9997

change_num = 0


def get_batch_images_seg(images,labels,batch_size,labels_nums,one_hot=False,shuffle=False,num_threads=1):
    '''
    :param images:图像
    :param labels:标签
    :param batch_size:
    :param labels_nums:标签个数
    :param one_hot:是否将labels转为one_hot的形式
    :param shuffle:是否打乱顺序,一般train时shuffle=True,验证时shuffle=False
    :return:返回batch的images和labels
    '''
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
    '''
    解析record文件:源文件的图像数据是RGB,uint8,[0,255],一般作为训练数据时,需要归一化到[0,1]
    :param filename:
    :param resize_height:
    :param resize_width:
    :param type:选择图像数据的返回类型
         None:默认将uint8-[0,255]转为float32-[0,255]
         normalization:归一化float32-[0,1]
         centralization:归一化float32-[0,1],再减均值中心化
    :return:
    '''
    # 创建文件队列,不限读取的数量
    filename_queue = tf.train.string_input_producer([filename])
    # create a reader from file queue
    reader = tf.TFRecordReader()
    # reader从文件队列中读入一个序列化的样本
    _, serialized_example = reader.read(filename_queue)
    # get feature from serialized example
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

    # # image = tf.decode_raw(features['image/format'], tf.uint8)
    # # label = tf.decode_raw(features['label/format'], tf.uint8)
    # label = tf.image.decode_image(tf.reshape(features['label/encoded'], shape=[]), 1)
    #
    # # PS:恢复原始图像数据,reshape的大小必须与保存之前的图像shape一致,否则出错
    # image = tf.reshape(image, [resize_height, resize_width, 3])  # 设置图像的维度
    # label = tf.reshape(label, [resize_height, resize_width, 1])
    #
    # # image = tf.cast(image, tf.float32) * (1. / 255.0)  # 归一化
    # image = tf.to_float(tf.image.convert_image_dtype(image, dtype=tf.uint8))
    # label = tf.to_int32(tf.image.convert_image_dtype(label, dtype=tf.uint8))
    #
    # image = tf.reshape(image, [resize_height, resize_width, 3])  # 设置图像的维度
    # label = tf.reshape(label, [resize_height, resize_width, 1])


    image = tf.image.decode_image(tf.reshape(features['image/encoded'], shape=[]), _DEPTH)
    image = tf.to_float(tf.image.convert_image_dtype(image, dtype=tf.uint8))
    image.set_shape([None, None, 3])

    label = tf.image.decode_image(tf.reshape(features['label/encoded'], shape=[]), 1)
    label = tf.to_int32(tf.image.convert_image_dtype(label, dtype=tf.uint8))
    label.set_shape([None, None, 1])

    # tf_image = tf.decode_raw(features['image_raw'], tf.uint8)  # 获得图像原始的数据
    #
    # tf_height = features['height']
    # tf_width = features['width']
    # tf_depth = features['depth']
    # tf_label = tf.cast(features['label'], tf.int32)
    #
    # tf_image = tf.reshape(tf_image, [resize_height, resize_width, 3])  # 设置图像的维度

    # image = tf.decode_raw(parsed['image/encoded'], tf.uint8)
    # image = tf.reshape(image, [resize_height, resize_width, 3])
    #
    # image = tf.cast(image, tf.float32) * (1. / 255.0)
    #
    # label = tf.decode_raw(parsed['label/encoded'], tf.uint8)
    #
    #
    # label = tf.reshape(label, [resize_height, resize_width, 1])

    image ,label = preprocess_image(image, label, True)

    return image,label

def preprocess_image(image, label, is_training):
  """Preprocess a single image of layout [height, width, depth]."""
  if is_training:
    # Randomly scale the image and label.
    image, label = preprocessing.random_rescale_image_and_label(
        image, label, _MIN_SCALE, _MAX_SCALE)

    # Randomly crop or pad a [_HEIGHT, _WIDTH] section of the image and label.
    image, label = preprocessing.random_crop_or_pad_image_and_label(
        image, label, _HEIGHT, _WIDTH, _IGNORE_LABEL)

    # Randomly flip the image and label horizontally.
    image, label = preprocessing.random_flip_left_right_image_and_label(
        image, label)

    image.set_shape([_HEIGHT, _WIDTH, 3])
    label.set_shape([_HEIGHT, _WIDTH, 1])
    print("seg11111111111",image,label)
  image = preprocessing.mean_image_subtraction(image)

  return image, label

def input_fn(is_training, data_dir, reid_data_dir= None,batch_size=16, num_epochs=1):
  """Input_fn using the tf.data input pipeline for CIFAR-10 dataset.

  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.

  Returns:
    A tuple of images and labels.
  """

  train_seg_record_file = os.path.join(data_dir,  'LIP_train5.record')
  # val_seg_record_file = os.path.join(data_dir,  'LIP_val5.record')

  train_images_seg, train_labels_seg = read_records_seg(train_seg_record_file, _HEIGHT, _WIDTH, type='normalization')
  train_images_batch_seg, train_labels_batch_seg = get_batch_images_seg(train_images_seg, train_labels_seg,
                                                            batch_size=batch_size, labels_nums=5,shuffle=True)

  # val_images_seg, val_labels_seg = read_records_seg(val_seg_record_file, _HEIGHT, _WIDTH, type='normalization')
  # val_images_batch_seg, val_labels_batch_seg = get_batch_images_seg(val_images_seg, val_labels_seg,
  #                                                       batch_size=batch_size, labels_nums=5,
  #                                                       one_hot=True, shuffle=False)

  images_seg  = train_images_batch_seg
  label_seg = train_labels_batch_seg

  train_record_file = os.path.join(reid_data_dir, 'train-512-170.tfrecords')
  # val_record_file = os.path.join(reid_data_dir, 'val-512-170.tfrecords')

  train_images, train_labels = read_records(train_record_file, _HEIGHT, _WIDTH, type='normalization')
  train_images_batch, train_labels_batch = get_batch_images(train_images, train_labels,
                                                            batch_size=batch_size, labels_nums=labels_nums,
                                                            one_hot=True, shuffle=True)
  print("reid2222222", train_images_batch.shape, train_labels_batch.shape)
  # val_images, val_labels = read_records(val_record_file, _HEIGHT, _WIDTH, type='normalization')
  # val_images_batch, val_labels_batch = get_batch_images(val_images, val_labels,
  #                                                       batch_size=batch_size, labels_nums=labels_nums,
  #                                                       one_hot=True, shuffle=False)

  images_reid  = train_images_batch
  label_reid = train_labels_batch

  # if is_training:
  #     images_reid  = train_images_batch
  #     label_reid = train_labels_batch
  # else:
  #     images_reid  = val_images_batch
  #     label_reid = val_labels_batch

  # if change_num % 2 ==0:
  #     images = images_seg
  #     labels =label_seg
  #     change_num =change_num+1
  # else:
  #     images =  images_reid
  #     labels =  label_reid
  #     change_num = change_num + 1

  images = {"seg": images_seg, "reid": images_reid}
  labels = {"seg": label_seg, "reid": label_reid}
  print (images,change_num)
  return images, labels


def main(unused_argv):
  # Using the Winograd non-fused algorithms provides a small performance boost.
  if FLAGS.clean_model_dir:
    shutil.rmtree(FLAGS.model_dir, ignore_errors=True)

  # session_config = tf.ConfigProto(log_device_placement=True)
  # session_config.gpu_options.per_process_gpu_memory_fraction = 0.5
  # run_config = tf.estimator.RunConfig().replace(session_config=session_config)
  # nn = tf.estimator.Estimator(model_fn=model_fn, config=run_config, params=model_params)

  # config = tf.ConfigProto()
  # config.gpu_options.allow_growth = True
  # session = tf.Session(config=config)

  session_config = tf.ConfigProto(
      log_device_placement=True,
      inter_op_parallelism_threads = 0,
      intra_op_parallelism_threads = 0,
      allow_soft_placement = True)
  session_config.gpu_options.allow_growth = True
  session_config.gpu_options.allocator_type = 'BFC'


  # Set up a RunConfig to only save checkpoints once per training cycle.
  run_config = tf.estimator.RunConfig().replace(session_config=session_config,save_checkpoints_steps = 1000,keep_checkpoint_max=10)##session_config=session_config,
  model = tf.estimator.Estimator(
      model_fn=deeplab_model.deeplabv3_plus_model_fn,
      model_dir=FLAGS.model_dir,
      config=run_config,
      params={
          'batch_size': FLAGS.batch_size,
          'pre_trained_model': FLAGS.pre_trained_model,
          'batch_norm_decay': _BATCH_NORM_DECAY,
          'num_classes': _NUM_CLASSES,
          'tensorboard_images_max_outputs': FLAGS.tensorboard_images_max_outputs,
          'momentum': _MOMENTUM,
          'freeze_batch_norm': FLAGS.freeze_batch_norm,
          'initial_global_step': FLAGS.initial_global_step
      })

  for _ in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
    tensors_to_log = {
      'step': 'global_step_',
      'learning_rate': 'learning_rate',
      'loss_seg': 'cross_entropy',
      'loss_reid': 'loss_reid',
      'train_px_acc': 'train_px_accuracy',
      'train_mean_iou': 'train_mean_iou',
      'accuracy_reid': 'accuracy_reid',
        'loss': 'loss',
    }


    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=100)
    train_hooks = [logging_hook]
    eval_hooks = None

    if FLAGS.debug:
      debug_hook = tf_debug.LocalCLIDebugHook()
      train_hooks.append(debug_hook)
      eval_hooks = [debug_hook]

    tf.logging.info("Start training.")
    model.train(
        input_fn=lambda: input_fn(True, FLAGS.data_dir,FLAGS.reid_data_dir, FLAGS.batch_size, FLAGS.epochs_per_eval),
        hooks=train_hooks,
        # steps=1  # For debug
    )


    # tf.logging.info("Start evaluation.")
    # # Evaluate the model and print results
    # eval_results = model.evaluate(
    #     # Batch size must be 1 for testing because the images' size differs
    #     input_fn=lambda: input_fn(False, FLAGS.data_dir, FLAGS.reid_data_dir,1),
    #     hooks=eval_hooks,
    #     # steps=1  # For debug
    # )
    # print(eval_results)


if __name__ == '__main__':
  tf.reset_default_graph()
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
