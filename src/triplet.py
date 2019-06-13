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

import time
import tensorflow as tf
import numpy as np
import os
import itertools
import facenet
from six.moves import xrange
import random
from log import logger
import scipy.misc


class Triplet(object):
    def __init__(self, ):
        self._round = 0       # get from different data source round-robin
        self.nrof_trainable_vars = len(tf.trainable_variables())
        self.gradient_buffer = [0.0] * self.nrof_trainable_vars
        self.nrof_gradients = 0

    def train(self, image_batch, args, sess, data_source, unsupervised, supervised_dataset, unsupervised_dataset, epoch,
              image_paths_placeholder, labels_placeholder, data_augmentations_placeholder,
              source_image_paths_placeholder, target_image_paths_placeholder, labels_batch,
              batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, enqueue_op, domain_enqueue_op,
              global_step, embeddings, loss, optimizer, gradients, gradient_placeholder, apply_gradient_op,
              summary_op, summary_writer, learning_rate_schedule_file, embedding_size,
              triplet_loss, domain_adaptation_loss, log_dir):
        if args.learning_rate > 0.0:
            lr = args.learning_rate
        else:
            lr = facenet.get_learning_rate_from_file(learning_rate_schedule_file, epoch)

        batch_number = 0
        while batch_number < args.epoch_size:
            # Sample people randomly from the dataset
            image_paths, num_per_class = self.sample_people(data_source, supervised_dataset, args.people_per_batch, args.images_per_person)

            logger.debug('Running forward pass on sampled images: ')
            start_time = time.time()
            nrof_examples = args.people_per_batch * args.images_per_person
            labels_array = np.reshape(np.arange(nrof_examples), (-1, 3))
            image_paths_array = np.reshape(np.expand_dims(np.array(image_paths), 1), (-1, 3))
            data_augmentations_array = self.get_data_augmentations_array(labels_array)
            sess.run(enqueue_op, feed_dict={image_paths_placeholder: image_paths_array,
                                            labels_placeholder: labels_array,
                                            data_augmentations_placeholder: data_augmentations_array})
            emb_array = np.zeros((nrof_examples, embedding_size))
            nrof_batches = int(np.ceil(nrof_examples / args.batch_size))
            for i in range(nrof_batches):
                batch_size = min(nrof_examples - i * args.batch_size, args.batch_size)
                emb, lab = sess.run([embeddings, labels_batch], feed_dict={batch_size_placeholder: batch_size,
                                                                           learning_rate_placeholder: lr,
                                                                           phase_train_placeholder: True})
                emb_array[lab, :] = emb

            logger.debug('train time: %.3f' % (time.time()-start_time))

            # Select triplets based on the embeddings
            logger.debug('Selecting suitable triplets for training')
            # triplets, nrof_random_negs, nrof_triplets = self.select_triplets(emb_array, num_per_class,
            #                                                             image_paths, args.people_per_batch, args.alpha)
            triplets, nrof_random_negs, nrof_triplets = self.select_triplets_ceiling(emb_array, num_per_class,
                                                                        image_paths, args.people_per_batch, args.alpha, args.max_triplet_per_select)
            selection_time = time.time() - start_time
            logger.debug('(nrof_random_negs, nrof_triplets) = (%d, %d): time=%.3f seconds' %
                  (nrof_random_negs, nrof_triplets, selection_time))

            # Perform training on the selected triplets
            nrof_batches = int(np.ceil(nrof_triplets * 3 / args.batch_size))
            triplet_paths = list(itertools.chain(*triplets))
            labels_array = np.reshape(np.arange(len(triplet_paths)), (-1, 3))
            triplet_paths_array = np.reshape(np.expand_dims(np.array(triplet_paths), 1), (-1, 3))
            data_augmentations_array = self.get_data_augmentations_array(labels_array)
            sess.run(enqueue_op, feed_dict={image_paths_placeholder: triplet_paths_array,
                                            labels_placeholder: labels_array,
                                            data_augmentations_placeholder: data_augmentations_array})
            if args.data_source == 'MULTIPLE' and args.unsupervised != 'NONE':
                source_image_paths, target_image_paths = self.sample_unsupervised_dataset(unsupervised_dataset,
                                                                                  nrof_batches * args.batch_size//2)
                logger.debug("before domain_enqueue_op")
                source_image_paths_array = np.expand_dims(np.array(source_image_paths), 1)
                target_image_paths_array = np.expand_dims(np.array(target_image_paths), 1)
                sess.run(domain_enqueue_op, {source_image_paths_placeholder: source_image_paths_array,
                                         target_image_paths_placeholder: target_image_paths_array})
                logger.debug("after domain_enqueue_op")
            nrof_examples = len(triplet_paths)
            train_time = 0
            i = 0
            emb_array = np.zeros((nrof_examples, embedding_size))
            loss_array = np.zeros((nrof_triplets,))
            summary = tf.Summary()
            step = 0
            while i < nrof_batches:
                start_time = time.time()
                batch_size = min(nrof_examples - i * args.batch_size, args.batch_size)
                feed_dict = {batch_size_placeholder: batch_size,
                             # round_placeholder: self._round,
                             learning_rate_placeholder: lr,
                             phase_train_placeholder: True}
                images, err, grads, step, emb, lab, triplet_loss_, domain_adaptation_loss_= sess.run(
                    [image_batch, loss, gradients, global_step, embeddings, labels_batch, triplet_loss, domain_adaptation_loss], feed_dict=feed_dict)
                # grads, _ = zip(*g_and_v)
                self.gradient_buffer = [sum(x) for x in zip(self.gradient_buffer, grads)]
                self.nrof_gradients += 1
                emb_array[lab, :] = emb
                loss_array[i] = err
                duration = time.time() - start_time
                # if args.random_flip:
                #     self.save_images(images, batch_number)
                logger.debug('Epoch [%d][%d/%d]\tRound [%d]\tgrad: [%d]\tTime %.3f\tLoss %f\ttriplet_Loss %f\tdomain_adaptation_loss %f' %
                      (epoch, batch_number + 1, args.epoch_size, self._round, self.nrof_gradients, duration, err, triplet_loss_, domain_adaptation_loss_))
                batch_number += 1
                i += 1
                train_time += duration
                summary.value.add(tag='loss', simple_value=err)

            # Add validation loss and accuracy to summary
            # pylint: disable=maybe-no-member
            summary.value.add(tag='time/selection', simple_value=selection_time)
            summary_writer.add_summary(summary, step)
            self._round += 1
            if self.nrof_gradients > 0:
            # if self.nrof_gradients > 0 and self._round % 3 == 0:
                logger.info("apply_gradient...")
                self.gradient_buffer = map(lambda x: x/self.nrof_gradients, self.gradient_buffer)
                feed_dict = {learning_rate_placeholder: lr}
                feed_dict_grad = {grad_placeholder: grad for grad_placeholder, grad in zip(gradient_placeholder, self.gradient_buffer)}
                feed_dict.update(feed_dict_grad)
                if args.cluster:
                    _, lr_ = sess.run([apply_gradient_op, optimizer._optimizer._lr], feed_dict=feed_dict)
                else:
                    _, lr_ = sess.run([apply_gradient_op, optimizer._lr], feed_dict=feed_dict)
                with open(os.path.join(log_dir, 'learing_rate.txt'), 'at') as f:
                    f.write('%d\t%.5f\n' % (epoch, lr_))
                self.gradient_buffer = [0.0] * self.nrof_trainable_vars
                self.nrof_gradients = 0
        return step

    def get_data_augmentations_array(self, labels_array):
        array_shape = labels_array.shape
        if self._round % 3 == 0:
            data_augmentations_array = np.random.randint(256, size=array_shape)
        else:
            data_augmentations_array = np.zeros_like(labels_array)
        return data_augmentations_array

    def save_images(self, images, batch_number):
        save_index = 0
        save_dir = os.path.join("/data/nfs/fengchen/tmp/aug", str(batch_number))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for image in images:
            save_path = os.path.join(save_dir, str(save_index) + '.png')
            scipy.misc.imsave(save_path, image)
            save_index += 1

    def select_triplets(self, embeddings, nrof_images_per_class, image_paths, people_per_batch, alpha):
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

    def select_triplets_ceiling(self, embeddings, nrof_images_per_class, image_paths, people_per_batch, alpha, max_triplet_per_select):
        """ Select the triplets for training
        """
        trip_idx = 0
        emb_start_idx = 0
        num_trips = 0
        triplets = []
        nrof_triplets = 0
        early_break = False

        for i in xrange(people_per_batch):
            if early_break:
                break
            nrof_images = int(nrof_images_per_class[i])
            for j in xrange(1,nrof_images):
                if early_break:
                    break
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
                        nrof_triplets += 1
                        trip_idx += 1

                    num_trips += 1
                    if nrof_triplets >= max_triplet_per_select:
                        logger.debug("reach max_triplet_per_select, break!")
                        early_break = True
                        break

            emb_start_idx += nrof_images

        if early_break:
            logger.debug("select_triplet break early")
        else:
            logger.debug("select_triplet stop normally")

        np.random.shuffle(triplets)
        return triplets, num_trips, len(triplets)

    def sample_people(self, data_source,  supervised_dataset, people_per_batch, images_per_person):
        nrof_images = people_per_batch * images_per_person
        if data_source == 'SINGLE':
            dataset = supervised_dataset
        else:
            dataset = self.list_people_compound(supervised_dataset)
            # dataset = self.list_people_round_robin(supervised_dataset)
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


    # from id, camera, id+camera compound
    def list_people_compound(self, supervised_dataset):
        return list(np.hstack(supervised_dataset.values()))


    # from id, camera, id+camera round-robin
    def list_people_round_robin(self, supervised_dataset):
        # dataset = []
        if self._round % 3 == 0:
            dataset = supervised_dataset['id']
        elif self._round % 3 == 1:
            dataset = supervised_dataset['camera']
        else:
            dataset = supervised_dataset['id+camera']

        logger.debug("round: %d, dataset len: %s", self._round, len(dataset))
        return dataset


    def sample_unsupervised_dataset(self, unsupervised_dataset, nrof_images):
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


    def triplet_loss(self, embeddings, embedding_size, alpha):
        # Split embeddings into anchor, positive and negative and calculate triplet loss
        anchor, positive, negative = tf.unstack(tf.reshape(embeddings, [-1, 3, embedding_size]), 3, 1)
        triplet_loss = facenet.triplet_loss(anchor, positive, negative, alpha)
        return triplet_loss