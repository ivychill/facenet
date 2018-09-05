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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
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
import itertools
import argparse
import facenet
import lfw
from mmd import *
 
from tensorflow.python.ops import data_flow_ops

from six.moves import xrange
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
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
    train_set_ID = facenet.get_dataset(args.data_dir_ID)
    train_set_camera = facenet.get_dataset(args.data_dir_camera)
    train_set_mix = facenet.get_dataset(args.data_dir_mix)
    
    print('Model directory: %s' % model_dir)
    print('Log directory: %s' % log_dir)
    if args.pretrained_model:
        print('Pre-trained model: %s' % os.path.expanduser(args.pretrained_model))
    
    if args.lfw_dir:
        print('LFW directory: %s' % args.lfw_dir)
        # Read the file containing the pairs used for testing
        pairs = lfw.read_pairs(os.path.expanduser(args.lfw_pairs))
        # Get the paths for the corresponding images
        lfw_paths, actual_issame = lfw.get_paths(os.path.expanduser(args.lfw_dir), pairs, args.lfw_file_ext)
        

    with tf.Graph().as_default():
        tf.set_random_seed(args.seed)
        global_step = tf.Variable(0, trainable=False)

        # Placeholder for the learning rate
        learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
        batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
        
        image_paths_placeholder_ID = tf.placeholder(tf.string, shape=(None,3), name='image_paths_ID')
        image_paths_placeholder_camera = tf.placeholder(tf.string, shape=(None, 3), name='image_paths_camera')
        image_paths_placeholder_mix = tf.placeholder(tf.string, shape=(None, 3), name='image_paths_mix')
        image_paths_placeholder_valid = tf.placeholder(tf.string, shape=(None, 3), name='image_paths_valid')
        labels_placeholder_ID = tf.placeholder(tf.int64, shape=(None,3), name='labels_ID')
        labels_placeholder_camera = tf.placeholder(tf.int64, shape=(None, 3), name='labels_camera')
        labels_placeholder_mix = tf.placeholder(tf.int64, shape=(None, 3), name='labels_mix')
        labels_placeholder_valid = tf.placeholder(tf.int64, shape=(None, 3), name='labels_valid')


        input_queue_ID = data_flow_ops.FIFOQueue(capacity=100000,
                                    dtypes=[tf.string, tf.int64],
                                    shapes=[(3,), (3,)],
                                    shared_name=None, name=None)
        input_queue_camera = data_flow_ops.FIFOQueue(capacity=100000,
                                              dtypes=[tf.string, tf.int64],
                                              shapes=[(3,), (3,)],
                                              shared_name=None, name=None)
        input_queue_mix = data_flow_ops.FIFOQueue(capacity=100000,
                                                     dtypes=[tf.string, tf.int64],
                                                     shapes=[(3,), (3,)],
                                                     shared_name=None, name=None)
        input_queue_valid = data_flow_ops.FIFOQueue(capacity=100000,
                                                     dtypes=[tf.string, tf.int64],
                                                     shapes=[(3,), (3,)],
                                                     shared_name=None, name=None)
        enqueue_op_ID = input_queue_ID.enqueue_many([image_paths_placeholder_ID, labels_placeholder_ID])
        enqueue_op_camera = input_queue_camera.enqueue_many([image_paths_placeholder_camera, labels_placeholder_camera])
        enqueue_op_mix = input_queue_mix.enqueue_many([image_paths_placeholder_mix, labels_placeholder_mix])
        enqueue_op_valid = input_queue_valid.enqueue_many([image_paths_placeholder_valid, labels_placeholder_valid])
        nrof_preprocess_threads = 4

        images_and_labels_ID = []
        images_and_labels_camera = []
        images_and_labels_mix = []
        images_and_labels_valid = []
        for _ in range(nrof_preprocess_threads):
            filenames, label = input_queue_ID.dequeue()
            images = []
            for filename in tf.unstack(filenames):
                file_contents = tf.read_file(filename)
                image = tf.image.decode_image(file_contents, channels=3)
                
                if args.random_crop:
                    image = tf.random_crop(image, [args.image_size, args.image_size, 3])
                else:
                    image = tf.image.resize_image_with_crop_or_pad(image, args.image_size, args.image_size)
                if args.random_flip:
                    image = tf.image.random_flip_left_right(image)
    
                #pylint: disable=no-member
                image.set_shape((args.image_size, args.image_size, 3))
                images.append(tf.image.per_image_standardization(image))
            images_and_labels_ID.append([images, label])

        for _ in range(nrof_preprocess_threads):
            filenames, label = input_queue_camera.dequeue()

            images = []
            for filename in tf.unstack(filenames):
                file_contents = tf.read_file(filename)
                image = tf.image.decode_image(file_contents, channels=3)

                if args.random_crop:
                    image = tf.random_crop(image, [args.image_size, args.image_size, 3])
                else:
                    image = tf.image.resize_image_with_crop_or_pad(image, args.image_size, args.image_size)
                if args.random_flip:
                    image = tf.image.random_flip_left_right(image)

                # pylint: disable=no-member
                image.set_shape((args.image_size, args.image_size, 3))
                images.append(tf.image.per_image_standardization(image))
            images_and_labels_camera.append([images, label])
        for _ in range(nrof_preprocess_threads):
            filenames, label = input_queue_mix.dequeue()

            images = []
            for filename in tf.unstack(filenames):
                file_contents = tf.read_file(filename)
                image = tf.image.decode_image(file_contents, channels=3)

                if args.random_crop:
                    image = tf.random_crop(image, [args.image_size, args.image_size, 3])
                else:
                    image = tf.image.resize_image_with_crop_or_pad(image, args.image_size, args.image_size)
                if args.random_flip:
                    image = tf.image.random_flip_left_right(image)

                # pylint: disable=no-member
                image.set_shape((args.image_size, args.image_size, 3))
                images.append(tf.image.per_image_standardization(image))
            images_and_labels_mix.append([images, label])
        for _ in range(nrof_preprocess_threads):
            filenames, label = input_queue_valid.dequeue()

            images = []
            for filename in tf.unstack(filenames):
                file_contents = tf.read_file(filename)
                image = tf.image.decode_image(file_contents, channels=3)

                if args.random_crop:
                    image = tf.random_crop(image, [args.image_size, args.image_size, 3])
                else:
                    image = tf.image.resize_image_with_crop_or_pad(image, args.image_size, args.image_size)
                if args.random_flip:
                    image = tf.image.random_flip_left_right(image)

                # pylint: disable=no-member
                image.set_shape((args.image_size, args.image_size, 3))
                images.append(tf.image.per_image_standardization(image))
            images_and_labels_valid.append([images, label])

        image_batch_ID, labels_batch_ID = tf.train.batch_join(
            images_and_labels_ID, batch_size=batch_size_placeholder,
            shapes=[(args.image_size, args.image_size, 3), ()], enqueue_many=True,
            capacity=4 * nrof_preprocess_threads * args.batch_size,
            allow_smaller_final_batch=True)
        image_batch_ID = tf.identity(image_batch_ID, 'image_batch_ID')
        image_batch_ID = tf.identity(image_batch_ID, 'input_ID')
        labels_batch_ID = tf.identity(labels_batch_ID, 'label_batch_ID')


        image_batch_camera, labels_batch_camera = tf.train.batch_join(
            images_and_labels_camera, batch_size=batch_size_placeholder,
            shapes=[(args.image_size, args.image_size, 3), ()], enqueue_many=True,
            capacity=4 * nrof_preprocess_threads * args.batch_size,
            allow_smaller_final_batch=True)
        image_batch_camera = tf.identity(image_batch_camera, 'image_batch_camera')
        image_batch_camera = tf.identity(image_batch_camera, 'input_camera')
        labels_batch_camera = tf.identity(labels_batch_camera, 'label_batch_camera')

        image_batch_mix, labels_batch_mix = tf.train.batch_join(
            images_and_labels_mix, batch_size=batch_size_placeholder,
            shapes=[(args.image_size, args.image_size, 3), ()], enqueue_many=True,
            capacity=4 * nrof_preprocess_threads * args.batch_size,
            allow_smaller_final_batch=True)
        image_batch_mix = tf.identity(image_batch_mix, 'image_batch_mix')
        image_batch_mix = tf.identity(image_batch_mix, 'input_mix')
        labels_batch_mix = tf.identity(labels_batch_mix, 'label_batch_mix')

        image_batch_valid, labels_batch_valid = tf.train.batch_join(
            images_and_labels_valid, batch_size=batch_size_placeholder,
            shapes=[(args.image_size, args.image_size, 3), ()], enqueue_many=True,
            capacity=4 * nrof_preprocess_threads * args.batch_size,
            allow_smaller_final_batch=True)
        image_batch_valid = tf.identity(image_batch_valid, 'image_batch_valid')
        labels_batch_valid = tf.identity(labels_batch_valid, 'label_batch_valid')

        # Build the inference graph
        prelogits_ID, feature_map1_ID, feature_map2_ID, feature_map3_ID, _ = network.inference(image_batch_ID, args.keep_probability,
            phase_train=phase_train_placeholder, bottleneck_layer_size=args.embedding_size,
            weight_decay=args.weight_decay)
        prelogits_camera, feature_map1_camera, feature_map2_camera, feature_map3_camera, _ = network.inference(image_batch_camera, args.keep_probability,
                                            phase_train=phase_train_placeholder,
                                            bottleneck_layer_size=args.embedding_size,
                                            weight_decay=args.weight_decay)
        prelogits_mix, _, _, _, _ = network.inference(
            image_batch_mix, args.keep_probability,
            phase_train=phase_train_placeholder,
            bottleneck_layer_size=args.embedding_size,
            weight_decay=args.weight_decay)

        prelogits_valid, _, _, _, _ = network.inference(image_batch_valid, args.keep_probability,
                                         phase_train=phase_train_placeholder, bottleneck_layer_size=args.embedding_size,
                                         weight_decay=args.weight_decay)

        embeddings_ID = tf.nn.l2_normalize(prelogits_ID, 1, 1e-10, name='embeddings_ID')
        embeddings_camera = tf.nn.l2_normalize(prelogits_camera, 1, 1e-10, name='embeddings_camera')
        embeddings_mix = tf.nn.l2_normalize(prelogits_mix, 1, 1e-10, name='embeddings_mix')
        embeddings_valid = tf.nn.l2_normalize(prelogits_valid, 1, 1e-10, name='embeddings_valid')

        # Split embeddings into anchor, positive and negative and calculate triplet loss
        anchor_ID, positive_ID, negative_ID = tf.unstack(tf.reshape(embeddings_ID, [-1,3,args.embedding_size]), 3, 1)
        triplet_loss_ID = facenet.triplet_loss(anchor_ID, positive_ID, negative_ID, args.alpha)

        anchor_camera, positive_camera, negative_camera = tf.unstack(tf.reshape(embeddings_camera, [-1, 3, args.embedding_size]), 3, 1)
        triplet_loss_camera = facenet.triplet_loss(anchor_camera, positive_camera, negative_camera, args.alpha)

        anchor_mix, positive_mix, negative_mix = tf.unstack(
            tf.reshape(embeddings_mix, [-1, 3, args.embedding_size]), 3, 1)
        triplet_loss_mix = facenet.triplet_loss(anchor_mix, positive_mix, negative_mix, args.alpha)

        learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
            args.learning_rate_decay_epochs*args.epoch_size, args.learning_rate_decay_factor, staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)

        # Calculate the total losses
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

        loss_feature_map1 = mmd_loss(feature_map1_ID, feature_map1_camera)
        loss_feature_map2 = mmd_loss(feature_map2_ID, feature_map2_camera)
        loss_feature_map3 = mmd_loss(feature_map3_ID, feature_map3_camera)

        loss_total = tf.add_n([triplet_loss_ID] + [triplet_loss_camera] + [triplet_loss_mix] + [loss_feature_map1] + [loss_feature_map2] + [loss_feature_map3] + regularization_losses, name='loss_total')

        # Build a Graph that trains the model with one batch of examples and updates the model parameters
        train_op = facenet.train(loss_total, global_step, args.optimizer,
            learning_rate, args.moving_average_decay, tf.global_variables())

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

            if args.pretrained_model:
                print('Restoring pretrained model: %s' % args.pretrained_model)
                saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)
                saver.restore(sess, os.path.expanduser(args.pretrained_model))
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)

            # Training and validation loop
            epoch = 0
            while epoch < args.max_nrof_epochs:
                step = sess.run(global_step, feed_dict=None)
                epoch = step // args.epoch_size
                # Train for one epoch
                train(args, sess, train_set_ID, train_set_camera, train_set_mix, epoch, image_paths_placeholder_ID, image_paths_placeholder_camera, image_paths_placeholder_mix, labels_placeholder_ID, labels_placeholder_camera, labels_placeholder_mix, labels_batch_ID, labels_batch_camera, labels_batch_mix,
                    batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, enqueue_op_ID,enqueue_op_camera, enqueue_op_mix, global_step,
                    embeddings_ID, embeddings_camera, embeddings_mix, loss_total, triplet_loss_ID, triplet_loss_camera, triplet_loss_mix, loss_feature_map1, loss_feature_map2, loss_feature_map3, regularization_losses, train_op, summary_writer, args.learning_rate_schedule_file,
                    args.embedding_size)

                # Save variables and the metagraph if it doesn't exist already
                save_variables_and_metagraph(sess, saver, summary_writer, model_dir, subdir, step)

                # Evaluate on LFW
                if args.lfw_dir:
                   evaluate(sess, lfw_paths, embeddings_valid, labels_batch_valid, image_paths_placeholder_valid, labels_placeholder_valid,
                           batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, enqueue_op_valid, actual_issame, args.batch_size,
                           args.lfw_nrof_folds, log_dir, step, summary_writer, args.embedding_size)

    return model_dir


def train(args, sess, dataset_ID, dataset_camera, dataset_mix, epoch, image_paths_placeholder_ID, image_paths_placeholder_camera, image_paths_placeholder_mix, labels_placeholder_ID, labels_placeholder_camera, labels_placeholder_mix, labels_batch_ID, labels_batch_camera, labels_batch_mix,
          batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, enqueue_op_ID, enqueue_op_camera, enqueue_op_mix, global_step,
          embeddings_ID, embeddings_camera, embeddings_mix, loss_total, triplet_loss_ID, triplet_loss_camera, triplet_loss_mix, loss_feature_map1, loss_feature_map2, loss_feature_map3, regularization_losses, train_op, summary_writer, learning_rate_schedule_file,
          embedding_size):
    batch_number = 0
    
    if args.learning_rate>0.0:
        lr = args.learning_rate
    else:
        lr = facenet.get_learning_rate_from_file(learning_rate_schedule_file, epoch)
    while batch_number < args.epoch_size:
        # Sample people randomly from the dataset
        image_paths_ID, num_per_class_ID = sample_people(dataset_ID, args.people_per_batch, args.images_per_person)
        image_paths_camera, num_per_class_camera = sample_people(dataset_camera, args.people_per_batch, args.images_per_person)
        image_paths_mix, num_per_class_mix = sample_people(dataset_mix, args.people_per_batch,
                                                                 args.images_per_person)
        print('Running forward pass on sampled images: ', end='')
        start_time = time.time()
        nrof_examples = args.people_per_batch * args.images_per_person
        labels_array_ID = np.reshape(np.arange(nrof_examples), (-1,3))
        labels_array_camera = np.reshape(np.arange(nrof_examples), (-1, 3))
        labels_array_mix = np.reshape(np.arange(nrof_examples), (-1, 3))

        image_paths_array_ID = np.reshape(np.expand_dims(np.array(image_paths_ID),1), (-1,3))
        image_paths_array_camera = np.reshape(np.expand_dims(np.array(image_paths_camera), 1), (-1, 3))
        image_paths_array_mix = np.reshape(np.expand_dims(np.array(image_paths_mix), 1), (-1, 3))

        sess.run(enqueue_op_ID, {image_paths_placeholder_ID: image_paths_array_ID, labels_placeholder_ID: labels_array_ID})
        sess.run(enqueue_op_camera, {image_paths_placeholder_camera: image_paths_array_camera, labels_placeholder_camera: labels_array_camera})
        sess.run(enqueue_op_mix, {image_paths_placeholder_mix: image_paths_array_mix,
                                     labels_placeholder_mix: labels_array_mix})
        emb_array_ID = np.zeros((nrof_examples, embedding_size))
        emb_array_camera = np.zeros((nrof_examples, embedding_size))
        emb_array_mix = np.zeros((nrof_examples, embedding_size))
        nrof_batches = int(np.ceil(nrof_examples / args.batch_size))
        for i in range(nrof_batches):
            batch_size = min(nrof_examples-i*args.batch_size, args.batch_size)

            emb_ID, lab_ID = sess.run([embeddings_ID, labels_batch_ID], feed_dict={batch_size_placeholder: batch_size,
                learning_rate_placeholder: lr, phase_train_placeholder: True})
            emb_array_ID[lab_ID, :] = emb_ID
            emb_camera, lab_camera = sess.run([embeddings_camera, labels_batch_camera], feed_dict={batch_size_placeholder: batch_size,
                learning_rate_placeholder: lr, phase_train_placeholder: True})
            emb_array_camera[lab_camera, :] = emb_camera
            emb_mix, lab_mix = sess.run([embeddings_mix, labels_batch_mix],
                                              feed_dict={batch_size_placeholder: batch_size,
                                                         learning_rate_placeholder: lr, phase_train_placeholder: True})
            emb_array_mix[lab_mix, :] = emb_mix


        print('%.3f' % (time.time()-start_time))

        # Select triplets based on the embeddings
        print('Selecting suitable triplets for training')
        triplets_ID, nrof_random_negs_ID, nrof_triplets_ID = select_triplets(emb_array_ID, num_per_class_ID,
            image_paths_ID, args.people_per_batch, args.alpha)
        triplets_camera, nrof_random_negs_camera, nrof_triplets_camera = select_triplets(emb_array_camera, num_per_class_camera,
            image_paths_camera, args.people_per_batch, args.alpha)
        triplets_mix, nrof_random_negs_mix, nrof_triplets_mix = select_triplets(emb_array_mix, num_per_class_mix,
                                                                                         image_paths_mix,
                                                                                         args.people_per_batch,
                                                                                         args.alpha)

        nrof_triplets = min(nrof_triplets_ID, nrof_triplets_camera, nrof_triplets_mix)
        nrof_triplets_ID = nrof_triplets
        nrof_triplets_camera = nrof_triplets
        nrof_triplets_mix = nrof_triplets
        triplets_ID_new = triplets_ID[0:nrof_triplets]
        triplets_camera_new = triplets_camera[0:nrof_triplets]
        triplets_mix_new = triplets_mix[0:nrof_triplets]

        selection_time = time.time() - start_time
        print('(nrof_random_negs, nrof_triplets) of ID = ( %d, %d): time=%.3f seconds' %
            (nrof_random_negs_ID, nrof_triplets_ID, selection_time))
        print('(nrof_random_negs, nrof_triplets) of camera = ( %d, %d): time=%.3f seconds' %
              (nrof_random_negs_camera, nrof_triplets_camera, selection_time))
        print('(nrof_random_negs, nrof_triplets) of mix = ( %d, %d): time=%.3f seconds' %
              (nrof_random_negs_mix, nrof_triplets_mix, selection_time))
        # Perform training on the selected triplets
        nrof_batches_ID = int(np.ceil(nrof_triplets_ID*3/args.batch_size))
        nrof_batches_camera = int(np.ceil(nrof_triplets_camera * 3 / args.batch_size))
        nrof_batches_mix = int(np.ceil(nrof_triplets_mix * 3 / args.batch_size))
        nrof_batches = min(nrof_batches_camera, nrof_batches_ID, nrof_batches_mix)

        triplet_paths_ID = list(itertools.chain(*triplets_ID_new))
        triplet_paths_camera = list(itertools.chain(*triplets_camera_new))
        triplet_paths_mix = list(itertools.chain(*triplets_mix_new))

        labels_array_ID = np.reshape(np.arange(len(triplet_paths_ID)),(-1,3))
        labels_array_camera = np.reshape(np.arange(len(triplet_paths_camera)), (-1, 3))
        labels_array_mix = np.reshape(np.arange(len(triplet_paths_mix)), (-1, 3))

        triplet_paths_array_ID = np.reshape(np.expand_dims(np.array(triplet_paths_ID),1), (-1,3))
        triplet_paths_array_camera = np.reshape(np.expand_dims(np.array(triplet_paths_camera), 1), (-1, 3))
        triplet_paths_array_mix = np.reshape(np.expand_dims(np.array(triplet_paths_mix), 1), (-1, 3))

        sess.run(enqueue_op_ID, {image_paths_placeholder_ID: triplet_paths_array_ID, labels_placeholder_ID: labels_array_ID})
        sess.run(enqueue_op_camera, {image_paths_placeholder_camera: triplet_paths_array_camera, labels_placeholder_camera: labels_array_camera})
        sess.run(enqueue_op_mix, {image_paths_placeholder_mix: triplet_paths_array_mix, labels_placeholder_mix: labels_array_mix})

        nrof_examples_ID = len(triplet_paths_ID)
        nrof_examples_camera = len(triplet_paths_camera)
        nrof_examples_mix = len(triplet_paths_mix)
        nrof_examples = min(nrof_examples_ID, nrof_examples_camera, nrof_examples_mix)
        train_time = 0
        i = 0

        summary = tf.Summary()
        while i < nrof_batches:
            start_time = time.time()
            batch_size = min(nrof_examples-i*args.batch_size, args.batch_size)
            feed_dict = {batch_size_placeholder: batch_size, learning_rate_placeholder: lr, phase_train_placeholder: True}
            # err_feature_map1 = sess.run([loss_feature_map1],feed_dict=feed_dict)
            err_total, err_ID, err_camera, err_mix, err_feature_map1, err_feature_map2, err_feature_map3, err_regularization, _, step = sess.run([loss_total, triplet_loss_ID, triplet_loss_camera, triplet_loss_mix, loss_feature_map1, loss_feature_map2, loss_feature_map3, regularization_losses, train_op,  global_step], feed_dict=feed_dict)

            duration = time.time() - start_time
            print('Epoch: [%d][%d/%d]\tTime %.3f\tTotal_Loss %2.3f' %
                  (epoch, batch_number+1, args.epoch_size, duration, err_total))
            print('triplet_loss_ID: %2.3f\ttriplet_loss_camera: %2.3f\ttriplet_loss_mix:%2.3f\tloss_feature_map1: %f\tloss_feature_map2:%f\tloss_feature_map3:%f' %
                  (err_ID, err_camera, err_mix, err_feature_map1, err_feature_map2, err_feature_map3))

            # print("regularization_losses is :", err_regularization)
            batch_number += 1
            i += 1
            train_time += duration
            summary.value.add(tag='loss', simple_value=err_total)

        # Add validation loss and accuracy to summary
        #pylint: disable=maybe-no-member
        summary.value.add(tag='time/selection', simple_value=selection_time)
        summary_writer.add_summary(summary)
    return step
  
def select_triplets(embeddings, nrof_images_per_class, image_paths, people_per_batch, alpha):
    """ Select the triplets for training
    """
    trip_idx = 0
    emb_start_idx = 0
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
            a_idx = emb_start_idx + j - 1
            neg_dists_sqr = np.sum(np.square(embeddings[a_idx] - embeddings), 1)
            for pair in xrange(j, nrof_images): # For every possible positive pair.
                p_idx = emb_start_idx + pair
                pos_dist_sqr = np.sum(np.square(embeddings[a_idx]-embeddings[p_idx]))
                neg_dists_sqr[emb_start_idx:emb_start_idx+nrof_images] = np.NaN
                #all_neg = np.where(np.logical_and(neg_dists_sqr-pos_dist_sqr<alpha, pos_dist_sqr<neg_dists_sqr))[0]  # FaceNet selection
                all_neg = np.where(neg_dists_sqr-pos_dist_sqr<alpha)[0] # VGG Face selecction
                nrof_random_negs = all_neg.shape[0]
                if nrof_random_negs>0:
                    rnd_idx = np.random.randint(nrof_random_negs)
                    n_idx = all_neg[rnd_idx]
                    triplets.append((image_paths[a_idx], image_paths[p_idx], image_paths[n_idx]))
                    #print('Triplet %d: (%d, %d, %d), pos_dist=%2.6f, neg_dist=%2.6f (%d, %d, %d, %d, %d)' % 
                    #    (trip_idx, a_idx, p_idx, n_idx, pos_dist_sqr, neg_dists_sqr[n_idx], nrof_random_negs, rnd_idx, i, j, emb_start_idx))
                    trip_idx += 1

                num_trips += 1

        emb_start_idx += nrof_images

    np.random.shuffle(triplets)
    return triplets, num_trips, len(triplets)

def sample_people(dataset, people_per_batch, images_per_person):
    nrof_images = people_per_batch * images_per_person

    # Sample classes from the dataset
    nrof_classes = len(dataset)
    class_indices = np.arange(nrof_classes)
    np.random.shuffle(class_indices)
    
    i = 0
    image_paths = []
    num_per_class = []
    sampled_class_indices = []
    # Sample images from these classes until we have enough
    while len(image_paths)<nrof_images:
        class_index = class_indices[i]
        nrof_images_in_class = len(dataset[class_index])
        image_indices = np.arange(nrof_images_in_class)
        np.random.shuffle(image_indices)
        nrof_images_from_class = min(nrof_images_in_class, images_per_person, nrof_images-len(image_paths))
        idx = image_indices[0:nrof_images_from_class]
        image_paths_for_class = [dataset[class_index].image_paths[j] for j in idx]
        sampled_class_indices += [class_index]*nrof_images_from_class
        image_paths += image_paths_for_class
        num_per_class.append(nrof_images_from_class)
        i+=1
  
    return image_paths, num_per_class

def evaluate(sess, image_paths, embeddings, labels_batch, image_paths_placeholder, labels_placeholder, 
        batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, enqueue_op, actual_issame, batch_size, 
        nrof_folds, log_dir, step, summary_writer, embedding_size):
    start_time = time.time()
    # Run forward pass to calculate embeddings
    print('Running forward pass on LFW images: ', end='')
    
    nrof_images = len(actual_issame)*2
    assert(len(image_paths)==nrof_images)
    labels_array = np.reshape(np.arange(nrof_images),(-1,3))
    image_paths_array = np.reshape(np.expand_dims(np.array(image_paths),1), (-1,3))
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})
    emb_array = np.zeros((nrof_images, embedding_size))
    nrof_batches = int(np.ceil(nrof_images / batch_size))
    label_check_array = np.zeros((nrof_images,))
    for i in xrange(nrof_batches):
        batch_size = min(nrof_images-i*batch_size, batch_size)
        emb, lab = sess.run([embeddings, labels_batch], feed_dict={batch_size_placeholder: batch_size,
            learning_rate_placeholder: 0.0, phase_train_placeholder: False})
        emb_array[lab,:] = emb
        label_check_array[lab] = 1
    print('%.3f' % (time.time()-start_time))
    
    assert(np.all(label_check_array==1))
    
    _, _, accuracy, val, val_std, far = lfw.evaluate(emb_array, actual_issame, nrof_folds=nrof_folds)
    
    print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
    lfw_time = time.time() - start_time
    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag='lfw/accuracy', simple_value=np.mean(accuracy))
    summary.value.add(tag='lfw/val_rate', simple_value=val)
    summary.value.add(tag='time/lfw', simple_value=lfw_time)
    summary_writer.add_summary(summary, step)
    with open(os.path.join(log_dir,'lfw_result.txt'),'at') as f:
        f.write('%d\t%.5f\t%.5f\n' % (step, np.mean(accuracy), val))

def save_variables_and_metagraph(sess, saver, summary_writer, model_dir, model_name, step):
    # Save the model checkpoint
    print('Saving variables')
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds' % save_time_variables)
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
    save_time_metagraph = 0  
    if not os.path.exists(metagraph_filename):
        print('Saving metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2f seconds' % save_time_metagraph)
    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag='time/save_variables', simple_value=save_time_variables)
    summary.value.add(tag='time/save_metagraph', simple_value=save_time_metagraph)
    summary_writer.add_summary(summary, step)
  
  
def get_learning_rate_from_file(filename, epoch):
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.split('#', 1)[0]
            if line:
                par = line.strip().split(':')
                e = int(par[0])
                lr = float(par[1])
                if e <= epoch:
                    learning_rate = lr
                else:
                    return learning_rate
    

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
    parser.add_argument('--data_dir_ID', type=str,
        help='Path to the data_ID directory containing aligned face patches.',
        default='~/datasets/casia/casia_maxpy_mtcnnalign_182_160')
    parser.add_argument('--data_dir_camera', type=str,
        help='Path to the data_camera directory containing aligned face patches.',
        default='~/datasets/casia/casia_maxpy_mtcnnalign_182_160')
    parser.add_argument('--data_dir_mix', type=str,
                        help='Path to the data_mix directory containing aligned face patches.',
                        default='~/datasets/casia/casia_maxpy_mtcnnalign_182_160')
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
