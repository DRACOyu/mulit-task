#coding=utf-8

import tensorflow as tf
import numpy as np
import pdb
import os
from datetime import datetime
import slim.nets.inception_v3 as inception_v3
from create_tf_record import *
import tensorflow.contrib.slim as slim
from scipy.spatial.distance import cdist
from sklearn.metrics import average_precision_score

from utils import preprocessing

_WEIGHT_DECAY = 5e-4
_NUM_CLASSES = 5
_MIN_SCALE = 0.5
_MAX_SCALE = 2.0
_IGNORE_LABEL = 255

labels_nums = 1041  # 类别个数
batch_size = 32  #
resize_height = 512 # 指定存储图片高度
resize_width = 170  # 指定存储图片宽度
depths = 3
data_shape = [batch_size, resize_height, resize_width, depths]



# 定义input_images为图片数据
input_images = tf.placeholder(dtype=tf.float32, shape=[None, resize_height, resize_width, depths], name='input')
# 定义input_labels为labels数据
# input_labels = tf.placeholder(dtype=tf.int32, shape=[None], name='label')
input_labels = tf.placeholder(dtype=tf.int32, shape=[None, labels_nums], name='label')
input_labels_seg = tf.placeholder(dtype=tf.int32, shape=[None,resize_height, resize_width,1], name='label_seg')


# 定义dropout的概率
keep_prob = tf.placeholder(tf.float32,name='keep_prob')
is_training = tf.placeholder(tf.bool, name='is_training')

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


def net_evaluation(sess,loss,accuracy,val_images_batch,val_labels_batch,val_nums):
    val_max_steps = int(val_nums / batch_size)
    val_losses = []
    val_accs = []
    for _ in range(val_max_steps):
        val_x, val_y = sess.run([val_images_batch, val_labels_batch])
        # print('labels:',val_y)
        # val_loss = sess.run(loss, feed_dict={x: val_x, y: val_y, keep_prob: 1.0})
        # val_acc = sess.run(accuracy,feed_dict={x: val_x, y: val_y, keep_prob: 1.0})
        val_loss,val_acc = sess.run([loss,accuracy], feed_dict={input_images: val_x, input_labels: val_y, keep_prob:1.0, is_training: False})
        val_losses.append(val_loss)
        val_accs.append(val_acc)
    mean_loss = np.array(val_losses, dtype=np.float32).mean()
    mean_acc = np.array(val_accs, dtype=np.float32).mean()
    return mean_loss, mean_acc

def cmc(distmat, query_ids=None, gallery_ids=None, query_cams=None, gallery_cams=None, topk=500,
        separate_camera_set=False, single_gallery_shot=False, first_match_break=False):
    m, n = distmat.shape
    # Fill up default values
    if query_ids is None:
        query_ids = np.arange(m)
    if gallery_ids is None:
        gallery_ids = np.arange(n)
    if query_cams is None:
        query_cams = np.zeros(m).astype(np.int32)
    if gallery_cams is None:
        gallery_cams = np.ones(n).astype(np.int32)
    # Ensure numpy array
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)
    query_cams = np.asarray(query_cams)
    gallery_cams = np.asarray(gallery_cams)
    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    # Compute CMC for each query
    ret = np.zeros(topk)
    num_valid_queries = 0
    for i in range(m):
        # Filter out the same id and same camera
        valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                 (gallery_cams[indices[i]] != query_cams[i]))
        if separate_camera_set:
            # Filter out samples from same camera
            valid &= (gallery_cams[indices[i]] != query_cams[i])
        if not np.any(matches[i, valid]):
            continue
        if single_gallery_shot:
            repeat = 10
            gids = gallery_ids[indices[i][valid]]
            inds = np.where(valid)[0]
            ids_dict = defaultdict(list)
            for j, x in zip(inds, gids):
                ids_dict[x].append(j)
        else:
            repeat = 1
        for _ in range(repeat):
            if single_gallery_shot:
                # Randomly choose one instance for each id
                sampled = (valid & _unique_sample(ids_dict, len(valid)))
                index = np.nonzero(matches[i, sampled])[0]
            else:
                index = np.nonzero(matches[i, valid])[0]
            delta = 1. / (len(index) * repeat)
            for j, k in enumerate(index):
                if k - j >= topk:
                    break
                if first_match_break:
                    ret[k - j] += 1
                    break
                ret[k - j] += delta
        num_valid_queries += 1
    if num_valid_queries == 0:
        raise RuntimeError("No valid query")
    return ret.cumsum() / num_valid_queries


def mean_ap(distmat, query_ids=None, gallery_ids=None, query_cams=None, gallery_cams=None):
    m, n = distmat.shape
    # Fill up default values
    if query_ids is None:
        query_ids = np.arange(m)
    if gallery_ids is None:
        gallery_ids = np.arange(n)
    if query_cams is None:
        query_cams = np.zeros(m).astype(np.int32)
    if gallery_cams is None:
        gallery_cams = np.ones(n).astype(np.int32)
    # Ensure numpy array
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)
    query_cams = np.asarray(query_cams)
    gallery_cams = np.asarray(gallery_cams)
    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    # Compute AP for each query
    aps = []
    for i in range(m):
        # Filter out the same id and same camera
        valid = ((gallery_ids[indices[i]] != query_ids[i]) | (gallery_cams[indices[i]] != query_cams[i]))
        # print(valid)
        y_true = matches[i, valid]
        y_score = -distmat[i][indices[i]][valid]
        if not np.any(y_true):
            continue
        aps.append(average_precision_score(y_true, y_score))
    if len(aps) == 0:
        raise RuntimeError("No valid query")

    # query_index = np.argwhere(gallery_ids==query_ids)
    # camera_index = np.argwhere(gallery_cams==query_cams)
    # good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    # junk_index1 = np.argwhere(gallery_ids==-1)
    # junk_index2 = np.intersect1d(query_index, camera_index)
    # junk_index = np.append(junk_index2, junk_index1) #.flatten())

    # CMC_tmp = compute_mAP(indices, good_index, junk_index)
    # return CMC_tmp
    return np.mean(aps)


def Evaluation(sess,net_reid_test):
    query_file = '/yzc/data/MSMT17_V1/msmt_list_query.txt'
    gallery_file = '/yzc/data/MSMT17_V1/msmt_list_gallery.txt'

    resize_height = 512  # 指定存储图片高度
    resize_width = 170  # 指定存储图片宽度

    query_feature = []
    query_label = []
    F = open(query_file).readlines()
    for f in F:
        image_path, label = f.split()
        image_path = os.path.join("/yzc/data/MSMT17_V1", image_path)
        im = read_image(image_path, resize_height, resize_width, normalization=True)
        im = im[np.newaxis, :]
        feature = sess.run(net_reid_test, feed_dict={input_images: im,is_training: False})

        query_feature.append(feature)
        query_label.append(label)

    gallery_feature = []
    gallery_label = []
    G = open(gallery_file).readlines()
    for f in G:
        image_path, label = f.split()
        image_path = os.path.join("/yzc/data/MSMT17_V1", image_path)
        im1 = read_image(image_path, resize_height, resize_width, normalization=True)
        im1 = im1[np.newaxis, :]
        feature1 = sess.run(net_reid_test, feed_dict={input_images: im1,is_training: False})

        gallery_feature.append(feature1)
        gallery_label.append(label)

    query_feature1 = np.asarray(query_feature)
    gallery_feature1 = np.asarray(gallery_feature)
    query_feature2 = tf.squeeze(query_feature1)
    gallery_feature2 = tf.squeeze(gallery_feature1)
    query_feature4 = query_feature2.eval()
    gallery_feature4 = gallery_feature2.eval()

    print("query_feature4", query_feature4.shape)
    print("gallery_feature4", gallery_feature4.shape)
    print("run  no error")

    dist = cdist(query_feature4, gallery_feature4)
    print(dist.shape)
    print(dist[0, :])
    r = cmc(dist, query_label, gallery_label,
            separate_camera_set=False,
            single_gallery_shot=False,
            first_match_break=True)

    gallery_label = np.array(gallery_label)
    query_label = np.array(query_label)
    m_ap = mean_ap(dist, query_label, gallery_label)
    print(m_ap)
    print(' mAP=%f' % (m_ap))
    print('mAP=%f, r@1=%f, r@10=%f, r@25=%f, r@50=%f, r@100=%f' % (
    m_ap, r[0], r[9], r[24], r[49], r[99]))
    return m_ap, r



def seg_loss_iou(logits,input_labels_seg):
    labels_nums_seg = 5
    pred_classes = tf.expand_dims(tf.argmax(logits, axis=3, output_type=tf.int32), axis=3)  # 16,512,170,1
    # train_var_list = [v for v in tf.trainable_variables()]
    # loss_seg_L2 = _WEIGHT_DECAY * tf.add_n([tf.nn.l2_loss(v) for v in train_var_list])

    labels_seg = tf.squeeze(input_labels_seg, axis=3)  # 16,512,170

    logits_by_num_classes = tf.reshape(logits, [-1, labels_nums_seg])  # ?,5
    labels_flat = tf.reshape(labels_seg, [-1, ])  # 1392640

    valid_indices = tf.to_int32(labels_flat <= labels_nums_seg - 1)  # 1392640
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
    loss_seg = tf.losses.sparse_softmax_cross_entropy(logits=valid_logits, labels=valid_labels)  ####+loss_seg_L2
    # loss_seg = tf.losses.get_total_loss(add_regularization_losses=True)  # 添加正则化损失loss=2.2
    return loss_seg,mean_iou


def Evaluation_simple(sess,net_reid_test):
    query_file = '/yzc/data/MSMT17_V1/msmt_list_query.txt'
    gallery_file = '/yzc/data/MSMT17_V1/msmt_list_gallery.txt'

    resize_height = 512  # 指定存储图片高度
    resize_width = 170  # 指定存储图片宽度

    query_feature = []
    query_label = []
    t=0
    t2=0
    F = open(query_file).readlines()
    for f in F:
        t=t+1
        if t <100:
            image_path, label = f.split()
            image_path = os.path.join("/yzc/data/MSMT17_V1", image_path)
            im = read_image(image_path, resize_height, resize_width, normalization=True)
            im = im[np.newaxis, :]
            feature = sess.run(net_reid_test, feed_dict={input_images: im,is_training: False})

            query_feature.append(feature)
            query_label.append(label)

    gallery_feature = []
    gallery_label = []
    G = open(gallery_file).readlines()
    for f in G:
        t2=t2+1
        if t2 <500:
            image_path, label = f.split()
            image_path = os.path.join("/yzc/data/MSMT17_V1", image_path)
            im1 = read_image(image_path, resize_height, resize_width, normalization=True)
            im1 = im1[np.newaxis, :]
            feature1 = sess.run(net_reid_test, feed_dict={input_images: im1,is_training: False})

            gallery_feature.append(feature1)
            gallery_label.append(label)

    query_feature1 = np.asarray(query_feature)
    gallery_feature1 = np.asarray(gallery_feature)
    query_feature2 = tf.squeeze(query_feature1)
    gallery_feature2 = tf.squeeze(gallery_feature1)
    query_feature4 = query_feature2.eval()
    gallery_feature4 = gallery_feature2.eval()

    print("query_feature4", query_feature4.shape)
    print("gallery_feature4", gallery_feature4.shape)
    print("run  no error")

    dist = cdist(query_feature4, gallery_feature4)
    print(dist.shape)
    print(dist[0, :])
    r = cmc(dist, query_label, gallery_label,
            separate_camera_set=False,
            single_gallery_shot=False,
            first_match_break=True)

    gallery_label = np.array(gallery_label)
    query_label = np.array(query_label)
    m_ap = mean_ap(dist, query_label, gallery_label)
    print(m_ap)
    print(' mAP=%f' % (m_ap))
    print('mAP=%f, r@1=%f, r@10=%f, r@25=%f, r@50=%f, r@100=%f' % (
    m_ap, r[0], r[9], r[24], r[49], r[99]))
    return m_ap, r


def train(train_record_file,
          data_dir,
          train_log_step,
          train_param,
          val_record_file,
          val_log_step,
          labels_nums,
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
    # train数据,训练数据一般要求打乱顺序shuffle=True
    train_images, train_labels = read_records(train_record_file, resize_height, resize_width, type='normalization')
    train_images_batch, train_labels_batch = get_batch_images(train_images, train_labels,
                                                              batch_size=batch_size, labels_nums=labels_nums,
                                                              one_hot=True, shuffle=True)
    # val数据,验证数据可以不需要打乱数据
    val_images, val_labels = read_records(val_record_file, resize_height, resize_width, type='normalization')
    val_images_batch, val_labels_batch = get_batch_images(val_images, val_labels,
                                                          batch_size=batch_size, labels_nums=labels_nums,
                                                          one_hot=True, shuffle=False)

    train_seg_record_file = os.path.join(data_dir, 'LIP_train5.record')
    val_seg_record_file = os.path.join(data_dir,  'LIP_val5.record')
    train_images_seg, train_labels_seg = read_records_seg(train_seg_record_file, resize_height, resize_width, type='normalization')
    train_images_batch_seg, train_labels_batch_seg = get_batch_images_seg(train_images_seg, train_labels_seg,
                                                                          batch_size=batch_size, labels_nums=5,
                                                                          shuffle=True)



    # Define the model:
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        out, end_points = inception_v3.inception_v3(inputs=input_images, num_classes=labels_nums, dropout_keep_prob=keep_prob, is_training=is_training)
    net_reid_test = end_points['AvgPool_1a']
    logits = end_points["decoder"]

    # logits = slim.conv2d(net, labels_nums, [1, 1], activation_fn=None,normalizer_fn=None, scope='label-out')
    # out = tf.squeeze(logits, [1, 2], name='SpatialSqueeze-1')
    # Specify the loss function: tf.losses定义的loss函数都会自动添加到loss函数,不需要add_loss()了
    tf.losses.softmax_cross_entropy(onehot_labels=input_labels, logits=out)#添加交叉熵损失loss=1.6
    # slim.losses.add_loss(my_loss)
    loss = tf.losses.get_total_loss(add_regularization_losses=True)#添加正则化损失loss=2.2
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(input_labels, 1)), tf.float32))


    loss_seg,train_mean_iou=seg_loss_iou(logits, input_labels_seg)


    # Specify the optimization scheme:
    global_step_ = tf.Variable(tf.constant(0))
    if (global_step_ + 1) % int(max_steps / 10) == 0:
        decay = ((1.0 - float(global_step_) / max_steps) ** 0.9)
        base_lr = base_lr * decay
    # base_lr= tf.train.exponential_decay(base_lr,global_step_,1000,decay)

    optimizer = tf.train.MomentumOptimizer(learning_rate=base_lr,momentum= 0.9)


    # global_step = tf.Variable(0, trainable=False)
    # learning_rate = tf.train.exponential_decay(0.05, global_step, 150, 0.9)
    #
    # optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
    # # train_tensor = optimizer.minimize(loss, global_step)
    # train_op = slim.learning.create_train_op(loss, optimizer,global_step=global_step)


    # 在定义训练的时候, 注意到我们使用了`batch_norm`层时,需要更新每一层的`average`和`variance`参数,
    # 更新的过程不包含在正常的训练过程中, 需要我们去手动像下面这样更新
    # 通过`tf.get_collection`获得所有需要更新的`op`
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if var.name.startswith('InceptionV3/decoder')]

    g_vars = [var for var in t_vars if var.name.startswith('InceptionV3/Logits')]

    # 使用`tensorflow`的控制流, 先执行更新算子, 再执行训练
    with tf.control_dependencies(update_ops):
        # create_train_op that ensures that when we evaluate it to get the loss,
        # the update_ops are done and the gradient updates are computed.
        # train_op = slim.learning.create_train_op(total_loss=loss,optimizer=optimizer)
        train_op = slim.learning.create_train_op(total_loss=loss, optimizer=optimizer,update_ops=g_vars)
        train_op_seg = slim.learning.create_train_op(total_loss=loss_seg, optimizer=optimizer,update_ops=d_vars)

    # Pretrained_model_dir = "/yzc/test/tensorflow-inceptionv3-reid-9/models/reid-9-imageft/model_ckpt-14w+-110000"
    Pretrained_model_dir = "/yzc/test/tensorflow-inceptionv3-reid-9/models/reid-9-imageft/model_ckpt-28w+.ckpt"
    # Pretrained_model_dir = "/yzc/test/test_module/models/sp-reid-ft/512/best_models_66000_0.9350.ckpt"
    # Pretrained_model_dir = "/yzc/test/test_module/models/sp-reid-ft/best_models_34000_0.9257.ckpt"
    # Pretrained_model_dir = "/yzc/test/inception_v3.ckpt"
    # exclusions = ['InceptionV3/Logits','InceptionV3/AuxLogits']

    exclusions = ['InceptionV3/AuxLogits', "InceptionV3/decoder",'InceptionV3/Logits']

    inception_except_logits = slim.get_variables_to_restore(exclude=exclusions)


    saver = tf.train.Saver(inception_except_logits)


    # inception_except_logits = slim.get_variables_to_restore(exclude=exclusions)
    # saver = tf.train.Saver(max_to_keep=0)
    max_acc = 0.0
    learning_data= []
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    with tf.Session() as sess:



        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # #这里的exclusions是不需要读取预训练模型中的Logits,因为默认的类别数目是1000，当你的类别数目不是1000的时候，如果还要读取的话，就会报错
        # #创建一个列表，包含除了exclusions之外所有需要读取的变量
        # inception_except_logits = slim.get_variables_to_restore()
        # #建立一个从预训练模型checkpoint中读取上述列表中的相应变量的参数的函数
        # init_fn = slim.assign_from_checkpoint_fn(Pretrained_model_dir, inception_except_logits,ignore_missing_vars=True)
        # #运行该函数
        # init_fn(sess)

        saver.restore(sess, Pretrained_model_dir)
        # saver.restore(sess, "/yzc/test/test_module/models/sp-reid-ft/best_models_34000_0.9257.ckpt")
        for i in range(max_steps + 1):
            batch_input_images, batch_input_labels = sess.run([train_images_batch, train_labels_batch])
            batch_input_images_seg, batch_input_labels_seg = sess.run([train_images_batch_seg, train_labels_batch_seg])

            if i %2 ==0:
                _, train_loss= sess.run([train_op, loss], feed_dict={input_images: batch_input_images,
                                                                         input_labels: batch_input_labels,
                                                                         global_step_: i,is_training: True})
            else:
                _, train_loss_seg= sess.run([train_op_seg, loss_seg], feed_dict={input_images: batch_input_images_seg,
                                                                     input_labels_seg: batch_input_labels_seg,
                                                                     global_step_: i,is_training: True})

            # _, train_loss= sess.run([train_op, loss], feed_dict={input_images: batch_input_images,
            #                                                              input_labels: batch_input_labels,
            #                                                              global_step_: i,is_training: True})

            # train_loss_seg = sess.run(loss_seg, feed_dict={input_images: batch_input_images_seg,
            #                                                             input_labels_seg: batch_input_labels_seg,
            #                                                             global_step_: i, is_training: True})

            # train测试(这里仅测试训练集的一个batch)global_step_: num_lr,
            if i % train_log_step == 0:
                train_acc = sess.run(accuracy, feed_dict={input_images: batch_input_images,
                                                          input_labels: batch_input_labels,
                                                          keep_prob: 1.0, is_training: False})
                train_acc_seg = sess.run(train_mean_iou, feed_dict={input_images: batch_input_images_seg,
                                                                input_labels_seg: batch_input_labels_seg,
                                                                global_step_: i, is_training: False})
                print("%s: Step [%d]  train Loss : %f, training accuracy :  %g  " % (
                datetime.now(), i, train_loss,train_acc))

            if i>=1000 and i%1100 ==0:
                print("--------------------------------eval----------------------------------------------")
                map,r  =  Evaluation_simple(sess,net_reid_test)
                print('mAP=%f, r@1=%f, r@10=%f, r@25=%f, r@50=%f, r@100=%f' % (
                    map, r[0], r[9], r[24], r[49], r[99]))

            # val测试(测试全部val数据)
            if i % val_log_step == 0:
                mean_loss, mean_acc = net_evaluation(sess, loss, accuracy, val_images_batch, val_labels_batch, val_nums)
                print("%s: Step [%d]  val Loss : %f, val accuracy :  %g" % (datetime.now(), i, mean_loss, mean_acc))
            # 模型保存:每迭代snapshot次或者最后一次保存模型
            if (i % snapshot == 0 and i > 0) or i == max_steps:
                print('-----save:{}-{}'.format(snapshot_prefix, i))
                saver.save(sess, snapshot_prefix, global_step=i)
            # 保存val准确率最高的模型
            # if mean_acc > max_acc and mean_acc > 0.93:
            #     max_acc = mean_acc
            #     path = os.path.dirname(snapshot_prefix)
            #     best_models = os.path.join(path, 'best_models_{}_{:.4f}.ckpt'.format(i, max_acc))
            #     print('------save:{}'.format(best_models))
            #     saver.save(sess, best_models)

        coord.request_stop()
        coord.join(threads)
    # 循环迭代过程


if __name__ == '__main__':
    train_record_file='/root/data/MSMT17_V1/record/train-512-170.tfrecords'
    val_record_file='/root/data/MSMT17_V1/record/val-512-170.tfrecords'
    data_dir= '/yzc/test/inceptionv3-msmt17-multi-dilated/dataset/'

    os.environ["CUDA_VISIBLE_DEVICES"] = "5"

    train_log_step=100
    base_lr = 0.01  # 学习率
    max_steps = 3500 # 迭代次数
    train_param=[base_lr,max_steps]

    val_log_step=100
    snapshot=4000#保存文件间隔
    snapshot_prefix='/yzc/test/test_module/models-base4/model'
    train(train_record_file=train_record_file,
          data_dir = data_dir,
          train_log_step=train_log_step,
          train_param=train_param,
          val_record_file=val_record_file,
          val_log_step=val_log_step,
          labels_nums=labels_nums,
          data_shape=data_shape,
          snapshot=snapshot,
          snapshot_prefix=snapshot_prefix)
