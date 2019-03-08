
# -*- coding:utf-8 -*-

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import random
import facenet
from backend import *
# from log_config import *

class Associative(object):
    def __init__(self, network, args):
        self.args = args
        NUM_LABELS = args.people_per_batch_assoc  # len(...)
        IMAGE_SHAPE = [args.image_size, args.image_size, 3]

        pairwise_path = args.data_dir_associative
        self.supervise_dataset = facenet.get_dataset(pairwise_path+"id")   # id
        self.unsupervise_dataset = facenet.get_dataset(pairwise_path+"camera")   # camera

        supervise_nrof_classes = len(self.supervise_dataset)
        unsupervise_nrof_classes = len(self.unsupervise_dataset)
        assert supervise_nrof_classes == unsupervise_nrof_classes
        assert args.people_per_batch_assoc <= supervise_nrof_classes

        self.model = SemisupModel(network.inference, NUM_LABELS, IMAGE_SHAPE, args.keep_probability)

    def loss(self):
        # sample array index instead of array, because index of supervise and unsupervise should be identical
        sampled_class_index = random.sample(np.arange(len(self.supervise_dataset)), self.args.people_per_batch_assoc)
        t_sup_images, t_sup_labels = self.sample_data(self.supervise_dataset, sampled_class_index)
        t_unsup_images, _ = self.sample_data(self.unsupervise_dataset, sampled_class_index)

        # Compute embeddings and logits.
        t_sup_emb, _, _, _, _ = self.model.image_to_embedding(t_sup_images, self.args.keep_probability, reuse=tf.AUTO_REUSE)
        t_unsup_emb, _, _, _, _ = self.model.image_to_embedding(t_unsup_images, self.args.keep_probability, reuse=tf.AUTO_REUSE)

        t_sup_emb = tf.nn.l2_normalize(t_sup_emb, 1, 1e-10)
        t_unsup_emb = tf.nn.l2_normalize(t_unsup_emb, 1, 1e-10)

        # Add losses.
        return self.model.add_semisup_loss(t_sup_emb, t_unsup_emb, t_sup_labels, visit_weight=1.0)

    def get_image_and_label(self):
        sampled_class_index = random.sample(np.arange(len(self.supervise_dataset)), self.args.people_per_batch_assoc)
        t_sup_images, t_sup_labels = self.sample_data(self.supervise_dataset, sampled_class_index)
        t_unsup_images, t_unsup_labels = self.sample_data(self.unsupervise_dataset, sampled_class_index)
        return t_sup_images, t_unsup_images, t_sup_labels, t_unsup_labels

    def get_image_and_label_dann(self):
        sampled_class_index = random.sample(np.arange(len(self.supervise_dataset)), self.args.people_per_batch_assoc)
        t_sup_images, t_unsup_images, t_sup_labels, t_unsup_labels = self.sample_data_dann(self.supervise_dataset, self.unsupervise_dataset, sampled_class_index)
        return t_sup_images, t_unsup_images, t_sup_labels, t_unsup_labels

    def sample_data_dann(self, dataset_source, dataset_target, sampled_class_index):
        # image_paths = []

        images_array_source = []
        images_array_target = []
        labels_array_source = []
        labels_array_target = []
        for label, index in enumerate(sampled_class_index):
            nrof_images_in_class_source = len(dataset_source[index])
            nrof_images_in_class_target = len(dataset_target[index])
            nrof_images_in_class = min(nrof_images_in_class_source, nrof_images_in_class_target)

            sampled_image_paths_source = random.sample(dataset_source[index].image_paths, min(nrof_images_in_class, self.args.images_per_person_assoc))
            sampled_image_paths_target = random.sample(dataset_target[index].image_paths, min(nrof_images_in_class, self.args.images_per_person_assoc))

            for path_source in sampled_image_paths_source:
                image_source = self.load_image(path_source)
                images_array_source.append(image_source)
                labels_array_source.append(label)

            for path_target in sampled_image_paths_target:
                image_target = self.load_image(path_target)
                images_array_target.append(image_target)
                labels_array_target.append(label)

        # logger.debug("labels_array: %s" % (labels_array))
        return images_array_source, images_array_target, labels_array_source, labels_array_target

    def sample_data(self, dataset, sampled_class_index):
        # image_paths = []

        images_array = []
        labels_array = []
        for label, index in enumerate(sampled_class_index):
            nrof_images_in_class = len(dataset[index])
            sampled_image_paths = None
            # logger.debug("label: %d, index: %d, nrof_images_in_class: %d" % (label, index, nrof_images_in_class))
            # logger.debug("dataset[index]: %s" % dataset[index].image_paths)
            if nrof_images_in_class > self.args.images_per_person_assoc:
                sampled_image_paths = random.sample(dataset[index].image_paths, self.args.images_per_person_assoc)
            else:
                sampled_image_paths = dataset[index].image_paths

            # logger.debug("sampled_image_paths: %s" % (sampled_image_paths))

            for path in sampled_image_paths:
                image = self.load_image(path)
                images_array.append(image)
                labels_array.append(label)

        # logger.debug("labels_array: %s" % (labels_array))
        return images_array, labels_array

    def load_image(self, path):
        file_contents = tf.read_file(path)
        image = tf.image.decode_image(file_contents, channels=3)

        if self.args.random_crop:
            image = tf.random_crop(image, [self.args.image_size, self.args.image_size, 3])
        else:
            image = tf.image.resize_image_with_crop_or_pad(image, self.args.image_size, self.args.image_size)
        if self.args.random_flip:
            image = tf.image.random_flip_left_right(image)

        image.set_shape((self.args.image_size, self.args.image_size, 3))
        return tf.image.per_image_standardization(image)