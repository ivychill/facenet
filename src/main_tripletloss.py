"""Training a face recognizer with TensorFlow based on the FaceNet paper
FaceNet: A Unified Embedding for Face Recognition and Clustering: http://arxiv.org/abs/1503.03832
"""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THEa
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
import sys
import tensorflow as tf
import numpy as np
import importlib
import argparse
from domain_separation import losses
from log import *
from tensorflow.python.ops import data_flow_ops
from six.moves import xrange
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate
import facenet
import lfw
from triplet import Triplet
# import tsne_viz
import dataset
import horovod.tensorflow as hvd

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.cluster:
        hvd.init()
        rank = hvd.rank()
        subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
        log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir+'_'+str(rank))
        if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
            os.makedirs(log_dir)
        set_logger(logger, log_dir)
        if rank == 0:
            model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)
            if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
                os.makedirs(model_dir)
            # Write arguments to a text file
            facenet.write_arguments_to_file(args, os.path.join(log_dir, 'arguments.txt'))
            # Store some git revision info in a text file in the log directory
            src_path,_ = os.path.split(os.path.realpath(__file__))
            facenet.store_revision_info(src_path, log_dir, ' '.join(sys.argv))
            np.random.seed(seed=rank)
            logger.info('Model directory: %s' % model_dir)
            logger.info('Log directory: %s' % log_dir)

    else:
        subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
        log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)
        if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
            os.makedirs(log_dir)
        set_logger(logger, log_dir)
        model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)
        if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
            os.makedirs(model_dir)
        # Write arguments to a text file
        facenet.write_arguments_to_file(args, os.path.join(log_dir, 'arguments.txt'))
        # Store some git revision info in a text file in the log directory
        src_path, _ = os.path.split(os.path.realpath(__file__))
        facenet.store_revision_info(src_path, log_dir, ' '.join(sys.argv))
        np.random.seed(seed=args.seed)
        logger.info('Model directory: %s' % model_dir)
        logger.info('Log directory: %s' % log_dir)

    logger.info("train_tripletloss......")
    # supervised_dataset, unsupervised_dataset = dataset.get_dataset(args.data_dir, args.data_source)
    supervised_dataset = dataset.get_supervised_dataset(args.data_dir, args.data_source)
    unsupervised_dataset = {}
    if args.data_source == 'MULTIPLE' and args.unsupervised != 'NONE':
        unsupervised_dataset = dataset.get_dataset(args.data_dir, args.data_source)
    # logger.debug("supervised_dataset: %s, unsupervised_dataset: %s" % (supervised_dataset, unsupervised_dataset))
    # logger.debug("supervised_dataset['id']: %d" % (len(supervised_dataset['id'])))
    # logger.debug("supervised_dataset['camera']: %d" % (len(supervised_dataset['camera'])))
    # logger.debug("supervised_dataset['id+camera']: %d" % (len(supervised_dataset['id+camera'])))

    if args.pretrained_model:
        logger.info('Pre-trained model: %s' % os.path.expanduser(args.pretrained_model))

    if args.lfw_dir:
        logger.info('LFW directory: %s' % args.lfw_dir)
        # Read the file containing the pairs used for testing
        pairs = lfw.read_pairs(os.path.expanduser(args.lfw_pairs))
        # Get the paths for the corresponding images
        lfw_paths, lfw_actual_issame = lfw.get_paths(os.path.expanduser(args.lfw_dir), pairs, args.lfw_file_ext)
        # dataset = facenet.get_dataset(args.lfw_dir)
        # lfw_image_list, lfw_label_list = facenet.get_image_paths_and_labels(dataset)
    if args.val_dir:
        logger.info('val directory: %s' % args.val_dir)
        val_lfw_pairs = lfw.read_pairs(os.path.expanduser(args.val_pairs))
        # Get the paths for the corresponding images
        val_paths, val_actual_issame = lfw.get_paths(os.path.expanduser(args.val_dir), val_lfw_pairs, args.lfw_file_ext)
        # dataset = facenet.get_dataset(args.val_dir)
        # val_image_list, val_label_list = facenet.get_image_paths_and_labels(dataset)
        # META_FILE = "face_meta.tsv"
        # path_for_metadata = os.path.join(log_dir, META_FILE)
        # with open(path_for_metadata, 'w') as f:
        #     f.write("Index\tLabel\n")
        #     for index, label in enumerate(val_label_list):
        #         f.write("%d\t%d\n" % (index, label))

    with tf.Graph().as_default():
        tf.set_random_seed(args.seed)
        global_step = tf.Variable(0, trainable=False, name='global_step')

        # Placeholder for the learning rate
        learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
        batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
        image_paths_placeholder = tf.placeholder(tf.string, shape=(None,3), name='image_paths')
        labels_placeholder = tf.placeholder(tf.int64, shape=(None,3), name='labels')
        source_image_paths_placeholder = tf.placeholder(tf.string, shape=(None, 1), name='source_image_paths')
        target_image_paths_placeholder = tf.placeholder(tf.string, shape=(None, 1), name='target_image_paths')

        input_queue = data_flow_ops.FIFOQueue(capacity=100000,
                                    dtypes=[tf.string, tf.int64],
                                    shapes=[(3,), (3,)],
                                    shared_name=None, name=None)
        enqueue_op = input_queue.enqueue_many([image_paths_placeholder, labels_placeholder])
        image_batch, labels_batch = dataset.create_input_pipeline(input_queue, args, batch_size_placeholder)
        image_batch = tf.identity(image_batch, 'image_batch')
        image_batch = tf.identity(image_batch, 'input')
        labels_batch = tf.identity(labels_batch, 'label_batch')
        network = importlib.import_module(args.model_def)
        prelogits, _ = network.inference(image_batch, args.keep_probability,
            phase_train=phase_train_placeholder, bottleneck_layer_size=args.embedding_size,
            weight_decay=args.weight_decay)
        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
        triplet = Triplet()
        triplet_loss = triplet.triplet_loss(embeddings, args.embedding_size, args.alpha)
        domain_adaptation_loss = tf.constant(0.0)
        domain_enqueue_op = None
        if args.data_source == 'MULTIPLE' and args.unsupervised != 'NONE':
            # begin: domain adaptation loss
            domain_input_queue = data_flow_ops.FIFOQueue(capacity=100000,
                                                         dtypes=[tf.string, tf.string],
                                                         shapes=[(1,), (1,)],
                                                         shared_name=None, name=None)
            domain_enqueue_op = domain_input_queue.enqueue_many(
                [source_image_paths_placeholder, target_image_paths_placeholder], name='domain_enqueue_op')
            source_image_batch, target_image_batch = dataset.create_domain_input_pipeline(domain_input_queue, args,
                                                                                  batch_size_placeholder)
            source_image_batch = tf.identity(source_image_batch, 'source_image_batch')
            target_image_batch = tf.identity(target_image_batch, 'target_image_batch')
            source_prelogits, source_end_points = network.inference(source_image_batch, args.keep_probability,
                                                     phase_train=phase_train_placeholder,
                                                     bottleneck_layer_size=args.embedding_size,
                                                     weight_decay=args.weight_decay, reuse=True)
            target_prelogits, target_end_points = network.inference(target_image_batch, args.keep_probability,
                                                     phase_train=phase_train_placeholder,
                                                     bottleneck_layer_size=args.embedding_size,
                                                     weight_decay=args.weight_decay, reuse=True)
            # multiplier 1.0 may not be the best
            if args.unsupervised == 'MMD':
                domain_adaptation_loss = 1.0 * losses.mmd_loss(source_end_points['PreLogitsFlatten'], target_end_points['PreLogitsFlatten'], 1.0)
            elif args.unsupervised == 'DANN':
                domain_adaptation_loss = 0.1 * losses.dann_loss(source_end_points['PreLogitsFlatten'], target_end_points['PreLogitsFlatten'], 1.0)
            tf.add_to_collection('losses', domain_adaptation_loss)
            # end: domain adaptation loss

        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n([triplet_loss] + [domain_adaptation_loss] + regularization_losses , name='total_loss')

        learning_rate = get_learning_rate(args, learning_rate_placeholder, global_step)
        tf.summary.scalar('learning_rate', learning_rate)
        # # Build a Graph that trains the model with one batch of examples and updates the model parameters
        # train_op = facenet.train(total_loss, global_step, args.optimizer,
        #     learning_rate, args.moving_average_decay, args.cluster, args.nrof_warmup_epochs)
        # split train into 2 parts: compute gradient and apply gradient
        trained_vars = tf.trainable_variables()
        opt = facenet.get_optimizer(args.optimizer, learning_rate, args.cluster)
        grads_and_vars = facenet.compute_gradients(opt, total_loss)
        # grads = facenet.compute_gradients_excluding_vars(total_loss)
        grads, vars = zip(*grads_and_vars)
        grads = [g if g is not None else tf.zeros_like(v)
                 for g, v in zip(grads, vars)]
        gradient_placeholder = [tf.placeholder(tf.float32) for _ in xrange(len(trained_vars))]
        apply_gradient_op = facenet.apply_gradients(opt, zip(gradient_placeholder, trained_vars), global_step, args.moving_average_decay)

        # Create a saver
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=10)
        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # Start running operations on the Graph.
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        if args.cluster:
            # Pin GPU to be used to process local rank (one GPU per process)
            config.gpu_options.visible_device_list = str(hvd.local_rank())
        sess = tf.Session(config=config)

        # Initialize variables
        sess.run(tf.global_variables_initializer(), feed_dict={phase_train_placeholder:True})
        sess.run(tf.local_variables_initializer(), feed_dict={phase_train_placeholder:True})
        if args.cluster:
            sess.run(hvd.broadcast_global_variables(0))

        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)

        with sess.as_default():
            epoch = 0
            if args.pretrained_model:
                logger.info('Restoring pretrained model: %s' % args.pretrained_model)
                saver.restore(sess, os.path.expanduser(args.pretrained_model))
                if args.lfw_dir:
                    evaluate(sess, lfw_paths, embeddings, labels_batch, image_paths_placeholder, labels_placeholder,
                             batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, enqueue_op, lfw_actual_issame,
                             args.batch_size, args.lfw_nrof_folds, log_dir, 'lfw_raw',epoch, summary_writer, args.embedding_size)
                if args.val_dir:
                    evaluate(sess, val_paths, embeddings, labels_batch, image_paths_placeholder, labels_placeholder,
                             batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, enqueue_op, val_actual_issame,
                             args.batch_size, args.lfw_nrof_folds, log_dir, 'val_raw',epoch, summary_writer, args.embedding_size)

            # Training and validation loop
            while epoch < args.max_nrof_epochs:
                step = sess.run(global_step, feed_dict=None)
                # epoch = step // args.epoch_size
                # Train for one epoch
                triplet.train(args, sess, args.data_source, args.unsupervised, supervised_dataset, unsupervised_dataset, epoch,
                    image_paths_placeholder, labels_placeholder, source_image_paths_placeholder, target_image_paths_placeholder, labels_batch,
                    batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, enqueue_op, domain_enqueue_op, global_step,
                    embeddings, total_loss, opt, grads, gradient_placeholder, apply_gradient_op,
                    summary_op, summary_writer, args.learning_rate_schedule_file, args.embedding_size,
                    triplet_loss, domain_adaptation_loss, log_dir)

                if (args.cluster and rank == 0) or not args.cluster:
                    # Save variables and the metagraph if it doesn't exist already
                    # save_variables_and_metagraph(sess, saver, summary_writer, model_dir, subdir, step)
                    save_variables_and_metagraph(sess, saver, summary_writer, model_dir, subdir, epoch)

                if args.lfw_dir:
                    evaluate(sess, lfw_paths, embeddings, labels_batch, image_paths_placeholder, labels_placeholder,
                             batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, enqueue_op, lfw_actual_issame,
                             args.batch_size, args.lfw_nrof_folds, log_dir, 'lfw', epoch, summary_writer, args.embedding_size)

                if args.val_dir:
                    evaluate(sess, val_paths, embeddings, labels_batch, image_paths_placeholder, labels_placeholder,
                             batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, enqueue_op, val_actual_issame,
                             args.batch_size, args.lfw_nrof_folds, log_dir, 'val', epoch, summary_writer, args.embedding_size)
                epoch += 1

    return model_dir


def get_learning_rate(args, learning_rate_placeholder, global_step):
    if args.nrof_warmup_epochs > 0:
        # TODO: There may be a bug because global_step is always 0
        first_decay_steps = args.epoch_size
        warm_steps = int(args.nrof_warmup_epochs * args.epoch_size)
        # first_decay_steps += warm_steps
        learning_rate = tf.train.cosine_decay_restarts(args.learning_rate, global_step - warm_steps,
                                                       first_decay_steps,
                                                       t_mul=1.1, m_mul=0.25, alpha=0.0001, name=None)
        warmup_lr = learning_rate_placeholder * tf.cast(global_step, tf.float32) / tf.cast(warm_steps, tf.float32)
        learning_rate = tf.cond(global_step < warm_steps,
                                lambda: warmup_lr, lambda: learning_rate)
    else:
        learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
                                                   args.learning_rate_decay_epochs * args.epoch_size,
                                                   args.learning_rate_decay_factor, staircase=True)
    # linear scaling
    # allreduce perform averaging instead of summing
    if args.cluster:
        learning_rate = learning_rate * hvd.size()
    return learning_rate


def evaluate(sess, image_paths, embeddings, labels_batch, image_paths_placeholder, labels_placeholder,
             batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, enqueue_op, actual_issame,
             batch_size, nrof_folds, log_dir, prefix, epoch, summary_writer, embedding_size):
    start_time = time.time()
    # Run forward pass to calculate embeddings
    print('Running forward pass on LFW images: ', end='')

    nrof_images = len(actual_issame) * 2
    assert (len(image_paths) == nrof_images)
    labels_array = np.reshape(np.arange(nrof_images), (-1, 3))
    image_paths_array = np.reshape(np.expand_dims(np.array(image_paths), 1), (-1, 3))
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})
    emb_array = np.zeros((nrof_images, embedding_size))
    nrof_batches = int(np.ceil(nrof_images / batch_size))
    label_check_array = np.zeros((nrof_images,))
    for i in xrange(nrof_batches):
        batch_size = min(nrof_images - i * batch_size, batch_size)
        emb, lab = sess.run([embeddings, labels_batch], feed_dict={batch_size_placeholder: batch_size,
                                                                   learning_rate_placeholder: 0.0,
                                                                   phase_train_placeholder: False})
        emb_array[lab, :] = emb
        label_check_array[lab] = 1
    logger.debug('%.3f' % (time.time() - start_time))

    assert (np.all(label_check_array == 1))

    tpr, fpr, accuracy, val, val_std, far = lfw.evaluate(emb_array, actual_issame, nrof_folds=nrof_folds)
    logger.debug('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
    logger.debug('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
    auc = metrics.auc(fpr, tpr)
    logger.debug('Area Under Curve (AUC): %1.3f' % auc)
    eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
    logger.debug('Equal Error Rate (EER): %1.3f' % eer)
    lfw_time = time.time() - start_time
    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    # pylint: disable=maybe-no-member
    summary.value.add(tag='lfw/accuracy', simple_value=np.mean(accuracy))
    summary.value.add(tag='lfw/val_rate', simple_value=val)
    summary.value.add(tag='time/lfw', simple_value=lfw_time)
    summary_writer.add_summary(summary, epoch)
    with open(os.path.join(log_dir, prefix+'_result.txt'),'at') as f:
        f.write('%d\t%.5f\t%.5f\n' % (epoch, np.mean(accuracy), val))


def save_variables_and_metagraph(sess, saver, summary_writer, model_dir, model_name, epoch):
    # Save the model checkpoint
    logger.info('Saving variables')
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=epoch, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    logger.info('Variables saved in %.2f seconds' % save_time_variables)
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
    save_time_metagraph = 0  
    if not os.path.exists(metagraph_filename):
        logger.info('Saving metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        logger.info('Metagraph saved in %.2f seconds' % save_time_metagraph)
    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag='time/save_variables', simple_value=save_time_variables)
    summary.value.add(tag='time/save_metagraph', simple_value=save_time_metagraph)
    summary_writer.add_summary(summary, epoch)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--logs_base_dir', type=str, 
        help='Directory where to write event logs.', default='~/logs/facenet')
    parser.add_argument('--models_base_dir', type=str,
        help='Directory where to write trained models and checkpoints.', default='~/models/facenet')
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--pretrained_model', type=str,
        help='Load a pretrained model before training starts.')
    parser.add_argument('--data_source', type=str, choices=['SINGLE', 'MULTIPLE'],
        help='whether or not there are subdirs under data_dir', default='SINGLE')
    parser.add_argument('--data_dir', type=str,
        help='Path to the data directory containing aligned face patches.',
        default='~/datasets/casia/casia_maxpy_mtcnnalign_182_160')
    parser.add_argument('--data_dir_camera', type=str,
        help='Path to the data_camera directory containing aligned face patches.',
        default='~/datasets/casia/casia_maxpy_mtcnnalign_182_160')
    parser.add_argument('--data_dir_associative', type=str,
        help='Path to the data directory for association.',
        default='/data/kaiyu/webface+7k/')
    parser.add_argument('--model_def', type=str,
        help='Model definition. Points to a module containing the definition of the inference graph.', default='models.inception_resnet_v1')
    parser.add_argument('--max_nrof_epochs', type=int,
        help='Number of epochs to run.', default=500)
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--people_per_batch', type=int,
        help='Number of people per batch.', default=30)
    parser.add_argument('--images_per_person', type=int,
        help='Number of images per person.', default=10)
    parser.add_argument('--people_per_batch_assoc', type=int,
        help='Number of people per batch.', default=30)
    parser.add_argument('--images_per_person_assoc', type=int,
        help='Number of images per person.', default=10)
    parser.add_argument('--epoch_size', type=int,
        help='Number of batches per epoch.', default=10000)
    parser.add_argument('--alpha', type=float,
        help='Positive to negative triplet distance margin.', default=0.2)
    parser.add_argument('--embedding_size', type=int,
        help='Dimensionality of the embedding.', default=128)
    parser.add_argument('--random_crop', 
        help='Performs random cropping of training images. If false, the center image_size pixels from the training images are used. ' +
         'If the size of the images in the data directory is equal to image_size no cropping is performed', action='store_true')
    parser.add_argument('--random_flip', 
        help='Performs random horizontal flipping of training images.', action='store_true')
    parser.add_argument('--keep_probability', type=float,
        help='Keep probability of dropout for the fully connected layer(s).', default=1.0)
    parser.add_argument('--unsupervised', type=str, choices=['NONE', 'MMD', 'DANN'],
        help='whether of not unsupervised loss is added', default='WITHOUT')
    parser.add_argument('--weight_decay', type=float,
        help='L2 weight regularization.', default=0.0)
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
        help='The optimization algorithm to use', default='ADAGRAD')
    parser.add_argument('--learning_rate', type=float,
        help='Initial learning rate. If set to a negative value a learning rate ' +
        'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.1)
    parser.add_argument('--learning_rate_decay_epochs', type=int,
        help='Number of epochs between learning rate decay.', default=100)
    parser.add_argument('--learning_rate_decay_factor', type=float,
        help='Learning rate decay factor.', default=1.0)
    parser.add_argument('--moving_average_decay', type=float,
        help='Exponential decay for tracking of training parameters.', default=0.9999)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
    parser.add_argument('--learning_rate_schedule_file', type=str,
        help='File containing the learning rate schedule that is used when learning_rate is set to to -1.', default='data/learning_rate_schedule.txt')
    parser.add_argument('--nrof_preprocess_threads', type=int,
        help='Number of preprocessing (data loading and augmentation) threads.', default=4)
    parser.add_argument('--lfw_pairs', type=str,
        help='The file containing the pairs to use for validation.', default='data/pairs.txt')
    parser.add_argument('--lfw_file_ext', type=str,
        help='The file extension for the LFW dataset.', default='png', choices=['jpg', 'png'])
    parser.add_argument('--lfw_dir', type=str,
        help='Path to the data directory containing aligned face patches.', default='')
    parser.add_argument('--lfw_nrof_folds', type=int,
        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    parser.add_argument('--val_pairs', type=str, default='/data/yanhong.jia/datasets/facenet/datasets_for_train/valid_24peo_3D+camera/pairs.txt',
        help='The file containing the pairs to use for validation.')
    parser.add_argument('--val_dir', type=str, default='/data/yanhong.jia/datasets/facenet/datasets_for_train/valid_24peo_3D+camera',
        help='Path to the data directory containing aligned face patches.')
    parser.add_argument('--gpu', type=str,
        help='GPU id', default='0')
    parser.add_argument('--cluster', type=bool,
        help='Whether or not Data Parallel with multi GPU', default=False)
    parser.add_argument('--nrof_warmup_epochs', type=int,
        help='Number of warmup epoch', default=0)
    return parser.parse_args(argv)
  

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))