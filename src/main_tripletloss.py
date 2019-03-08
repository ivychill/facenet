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
from log_config import logger
from tensorflow.python.ops import data_flow_ops
from six.moves import xrange
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate
import facenet
import lfw
import run_tripletloss
import tsne_viz
import dataset
import random


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def main(args):
  
    network = importlib.import_module(args.model_def)

    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)

    # Write arguments to a text file
    facenet.write_arguments_to_file(args, os.path.join(log_dir, 'arguments.txt'))
        
    # Store some git revision info in a text file in the log directory
    src_path,_ = os.path.split(os.path.realpath(__file__))
    facenet.store_revision_info(src_path, log_dir, ' '.join(sys.argv))

    np.random.seed(seed=args.seed)
    supervised_dataset, unsupervised_dataset = get_dataset(args.data_dir)
    
    logger.info('Model directory: %s' % model_dir)
    logger.info('Log directory: %s' % log_dir)
    if args.pretrained_model:
        logger.info('Pre-trained model: %s' % os.path.expanduser(args.pretrained_model))
    
    if args.lfw_dir:
        print('LFW directory: %s' % args.lfw_dir)
        # Read the file containing the pairs used for testing
        pairs = lfw.read_pairs(os.path.expanduser(args.lfw_pairs))
        # Get the paths for the corresponding images
        lfw_paths, lfw_actual_issame = lfw.get_paths(os.path.expanduser(args.lfw_dir), pairs)
        # dataset = facenet.get_dataset(args.lfw_dir)
        # lfw_image_list, lfw_label_list = facenet.get_image_paths_and_labels(dataset)
    if args.val_dir:
        print('val directory: %s' % args.val_dir)
        val_lfw_pairs = lfw.read_pairs(os.path.expanduser(args.val_pairs))
        # Get the paths for the corresponding images
        val_paths, val_actual_issame = lfw.get_paths(os.path.expanduser(args.val_dir), val_lfw_pairs)
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
        global_step = tf.Variable(0, trainable=False)

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

        domain_input_queue = data_flow_ops.FIFOQueue(capacity=1000000,
                                                     dtypes=[tf.string, tf.string],
                                                     shapes=[(1,), (1,)],
                                                     shared_name=None, name=None)
        domain_enqueue_op = domain_input_queue.enqueue_many(
            [source_image_paths_placeholder, target_image_paths_placeholder], name='domain_enqueue_op')
        source_image_batch, target_image_batch = dataset.create_domain_input_pipeline(domain_input_queue, args,
                                                                              batch_size_placeholder)
        source_image_batch = tf.identity(source_image_batch, 'source_image_batch')
        target_image_batch = tf.identity(target_image_batch, 'target_image_batch')

        # Build the inference graph
        prelogits, _, _, _, _ = network.inference(image_batch, args.keep_probability,
            phase_train=phase_train_placeholder, bottleneck_layer_size=args.embedding_size,
            weight_decay=args.weight_decay)
        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
        # Split embeddings into anchor, positive and negative and calculate triplet loss
        anchor, positive, negative = tf.unstack(tf.reshape(embeddings, [-1,3,args.embedding_size]), 3, 1)
        triplet_loss = facenet.triplet_loss(anchor, positive, negative, args.alpha)

        # begin: domain adaptation loss
        _, source_end_points = network.inference(source_image_batch, args.keep_probability,
                                                 phase_train=phase_train_placeholder,
                                                 bottleneck_layer_size=args.embedding_size,
                                                 weight_decay=args.weight_decay, reuse=True)
        _, target_end_points = network.inference(target_image_batch, args.keep_probability,
                                                 phase_train=phase_train_placeholder,
                                                 bottleneck_layer_size=args.embedding_size,
                                                 weight_decay=args.weight_decay, reuse=True)
        # multiplier 1.0 may not be the best
        domain_adaptation_loss = 1.0 * losses.mmd_loss(source_end_points['PreLogitsFlatten'], target_end_points['PreLogitsFlatten'])
        tf.add_to_collection('losses', domain_adaptation_loss)
        # end: domain adaptation loss

        learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
            args.learning_rate_decay_epochs*args.epoch_size, args.learning_rate_decay_factor, staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)

        # Calculate the total losses
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

        total_loss = tf.add_n([triplet_loss] + [domain_adaptation_loss] + regularization_losses , name='total_loss')

        # Build a Graph that trains the model with one batch of examples and updates the model parameters
        train_op = facenet.train(total_loss, global_step, args.optimizer,
            learning_rate, args.moving_average_decay, tf.global_variables())

        # Create a saver
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=10)
        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # Start running operations on the Graph.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        # Initialize variables
        sess.run(tf.global_variables_initializer(), feed_dict={phase_train_placeholder:True})
        sess.run(tf.local_variables_initializer(), feed_dict={phase_train_placeholder:True})

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
                             batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, enqueue_op,
                             lfw_actual_issame, args.batch_size,
                             args.lfw_nrof_folds, log_dir, 'lfw_raw',epoch, summary_writer, args.embedding_size)
                if args.val_dir:
                    evaluate(sess, val_paths, embeddings, labels_batch, image_paths_placeholder, labels_placeholder,
                             batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, enqueue_op,
                             val_actual_issame, args.batch_size,
                             args.lfw_nrof_folds, log_dir, 'val_raw',epoch, summary_writer, args.embedding_size)

            # Training and validation loop

            while epoch < args.max_nrof_epochs:
                step = sess.run(global_step, feed_dict=None)
                epoch = step // args.epoch_size
                # Train for one epoch
                run_tripletloss.train(args, sess, supervised_dataset, unsupervised_dataset, epoch,
                      image_paths_placeholder, labels_placeholder, source_image_paths_placeholder, target_image_paths_placeholder, labels_batch,
                      batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, enqueue_op, domain_enqueue_op, global_step,
                      embeddings, total_loss, train_op, summary_op, summary_writer, args.learning_rate_schedule_file,
                      args.embedding_size, triplet_loss, domain_adaptation_loss)

                # Save variables and the metagraph if it doesn't exist already
                save_variables_and_metagraph(sess, saver, summary_writer, model_dir, subdir, step)

                if args.lfw_dir:
                    evaluate(sess, lfw_paths, embeddings, labels_batch, image_paths_placeholder, labels_placeholder,
                             batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, enqueue_op, lfw_actual_issame,
                             args.batch_size, args.lfw_nrof_folds, log_dir, 'lfw', epoch, summary_writer, args.embedding_size)

                if args.val_dir:
                    evaluate(sess, val_paths, embeddings, labels_batch, image_paths_placeholder, labels_placeholder,
                             batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, enqueue_op, val_actual_issame,
                             args.batch_size, args.lfw_nrof_folds, log_dir, 'val',epoch, summary_writer, args.embedding_size)

    return model_dir

  
def select_triplets(embeddings, nrof_images_per_class, image_paths, people_per_batch, alpha):
    """ Select the triplets for training
    """
    tripx = 0
    emb_startx = 0
    num_trips = 0
    triplets = []
    
    # VGG Face: Choosing good triplets is crucial and should strike a balance between
    #  selecting informative (i.e. challenging) examples and swamping training with examples that
    #  are too hard. This is achieve by extending each pair (a, p) to a triplet (a, p, n) by sampling
    #  the image n at random, but only between the ones that violate the triplet loss margin. The
    #  latter is a form of hard-negative mining, but it is not as aggressive (and much cheaper) than
    #  choosing the maximally violating example, as often done in structured output learning.

    for i in xrange(people_per_batch):
        nrof_images = int(nrof_images_per_class[i])
        for j in xrange(1,nrof_images):
            ax = emb_startx + j - 1
            neg_dists_sqr = np.sum(np.square(embeddings[ax] - embeddings), 1)
            for pair in xrange(j, nrof_images): # For every possible positive pair.
                px = emb_startx + pair
                pos_dist_sqr = np.sum(np.square(embeddings[ax]-embeddings[px]))
                neg_dists_sqr[emb_startx:emb_startx+nrof_images] = np.NaN
                #all_neg = np.where(np.logical_and(neg_dists_sqr-pos_dist_sqr<alpha, pos_dist_sqr<neg_dists_sqr))[0]  # FaceNet selection
                all_neg = np.where(neg_dists_sqr-pos_dist_sqr<alpha)[0] # VGG Face selecction
                nrof_random_negs = all_neg.shape[0]
                if nrof_random_negs>0:
                    rndx = np.random.randint(nrof_random_negs)
                    nx = all_neg[rndx]
                    triplets.append((image_paths[ax], image_paths[px], image_paths[nx]))
                    #logger.debug('Triplet %d: (%d, %d, %d), pos_dist=%2.6f, neg_dist=%2.6f (%d, %d, %d, %d, %d)' %
                    #    (tripx, ax, px, nx, pos_dist_sqr, neg_dists_sqr[nx], nrof_random_negs, rndx, i, j, emb_startx))
                    tripx += 1

                num_trips += 1

        emb_startx += nrof_images

    np.random.shuffle(triplets)
    return triplets, num_trips, len(triplets)


def sample_people(supervised_dataset, people_per_batch, images_per_person):
    nrof_images = people_per_batch * images_per_person
    # TODO:
    dataset = list(np.hstack(supervised_dataset.values()))
    # Sample classes from the dataset
    nrof_classes = len(dataset)
    class_indices = np.arange(nrof_classes)
    np.random.shuffle(class_indices)

    i = 0
    image_paths = []
    num_per_class = []
    sampled_class_indices = []
    # Sample images from these classes until we have enough
    while len(image_paths) < nrof_images:
        class_index = class_indices[i]
        nrof_images_in_class = len(dataset[class_index])
        image_indices = np.arange(nrof_images_in_class)
        np.random.shuffle(image_indices)
        nrof_images_from_class = min(nrof_images_in_class, images_per_person, nrof_images - len(image_paths))
        idx = image_indices[0:nrof_images_from_class]
        image_paths_for_class = [dataset[class_index].image_paths[j] for j in idx]
        sampled_class_indices += [class_index] * nrof_images_from_class
        image_paths += image_paths_for_class
        num_per_class.append(nrof_images_from_class)
        i += 1

    return image_paths, num_per_class


def sample_unsupervised_dataset(unsupervised_dataset, nrof_images):
    dataset = unsupervised_dataset

    image_paths = {}
    domains = list(dataset.keys())
    random.shuffle(domains)
    domains = domains[0:2]
    d = 0
    for domain in domains:
        random.shuffle(dataset[domain].image_paths)
        i = 0
        image_path = []
        while len(image_path) < nrof_images:
            image_path.append(dataset[domain].image_paths[i])
            i = (i + 1) % len(dataset[domain].image_paths)
        image_paths[d] = image_path
        d += 1

    return image_paths[0], image_paths[1]


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
    print('%.3f' % (time.time() - start_time))

    assert (np.all(label_check_array == 1))

    tpr, fpr, accuracy, val, val_std, far,eval = lfw.evaluate(emb_array, actual_issame, nrof_folds=nrof_folds)
    [tprs, fprs, roc_best_threshold, tprs_fold, fprs_fold, val_fold, far_fold, val_best_threshold] = eval
    print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
    auc = metrics.auc(fpr, tpr)
    print('Area Under Curve (AUC): %1.3f' % auc)
    eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
    print('Equal Error Rate (EER): %1.3f' % eer)
    lfw_time = time.time() - start_time
    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    # pylint: disable=maybe-no-member
    summary.value.add(tag='lfw/accuracy', simple_value=np.mean(accuracy))
    summary.value.add(tag='lfw/val_rate', simple_value=val)
    summary.value.add(tag='time/lfw', simple_value=lfw_time)
    summary_writer.add_summary(summary, epoch)
    with open(os.path.join(log_dir, prefix +'_result.csv'), 'at') as f:
        f.write('epoch,fold, roc_threshold, accuracy,tpr,fpr,val_threshold,val,far\n')
        for flod_id in range(nrof_folds):
            f.write('%d,%d,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f\n' % (epoch,flod_id,
            roc_best_threshold[flod_id], accuracy[flod_id],tprs_fold[flod_id],fprs_fold[flod_id],
            val_best_threshold[flod_id],val_fold[flod_id], far_fold[flod_id]))
        f.write('%d,mean,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f\n' % (epoch,
                                                                np.mean(roc_best_threshold), np.mean(accuracy),
                                                                np.mean(tprs_fold), np.mean(fprs_fold),
                                                                np.mean(val_best_threshold), np.mean(val_fold),
                                                                np.mean(far_fold)))

    tsne_viz.log_evaluate(log_dir, np.mean(accuracy), np.std(accuracy), val, val_std, far, auc, eer, prefix, epoch)
    tsne_viz.plotFaceROC(log_dir,prefix+'_'+str(epoch),tprs, fprs,nrof_folds)

    embeddings1 = emb_array[0::2]
    embeddings2 = emb_array[1::2]
    tsne_viz.plotExp(log_dir,
                     facenet.distance(embeddings1,embeddings2,distance_metric=0),
                     facenet.distance(embeddings1,embeddings2,distance_metric=1),
                     np.asarray(actual_issame),
                     prefix + '_' + str(epoch))
    return emb_array


def save_variables_and_metagraph(sess, saver, summary_writer, model_dir, model_name, step):
    # Save the model checkpoint
    logger.info('Saving variables')
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
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
    summary_writer.add_summary(summary, step)
    

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
        help='Number of batches per epoch.', default=1000)
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

    # Parameters for validation on LFW
    parser.add_argument('--lfw_pairs', type=str,
        help='The file containing the pairs to use for validation.', default='data/pairs.txt')
    parser.add_argument('--lfw_file_ext', type=str,
        help='The file extension for the LFW dataset.', default='png', choices=['jpg', 'png'])
    parser.add_argument('--lfw_dir', type=str,
        help='Path to the data directory containing aligned face patches.', default='')
    parser.add_argument('--lfw_nrof_folds', type=int,
        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    return parser.parse_args(argv)
  

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))