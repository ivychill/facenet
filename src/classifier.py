"""An example of how to use your own dataset to train a classifier that recognizes people.
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

import tensorflow as tf
import numpy as np
import argparse
import facenet
import os
import sys
import math
import pickle
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
# from sklearn.lda import LDA
from sklearn.preprocessing import LabelEncoder
from sklearn.grid_search import GridSearchCV
from sklearn.mixture import GMM
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from datetime import datetime
import time


def get_image_paths_and_labels(dataset):
    image_paths_flat = []
    class_labels_flat = []
    labels = []
    for i in range(len(dataset)):
        image_paths_flat += dataset[i].image_paths
        labels += [i] * len(dataset[i].image_paths)
        class_labels_flat += [dataset[i].name.replace('_', ' ')] * len(dataset[i].image_paths)
    return image_paths_flat, labels, class_labels_flat


def classifier(args, args_mode, dataset, sess):
    # Check that there are at least one training image per class
    for cls in dataset:
        # print(cls.name,'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        if (len(cls.image_paths) < 1):
            print(cls.image_paths, "@@@@@@@@@@@@@@@@@@@@@@@")
        assert (len(cls.image_paths) > 0, 'There must be at least one image for each class in the dataset')

    paths, labels, class_labels = get_image_paths_and_labels(dataset)

    print('Number of classes: %d' % len(dataset))
    print('Number of images: %d' % len(paths))

    # Get input and output tensors
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    embedding_size = embeddings.get_shape()[1]

    # Run forward pass to calculate embeddings
    print('Calculating features for images')
    nrof_images = len(paths)
    nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / args.batch_size))
    emb_array = np.zeros((nrof_images, embedding_size))
    for i in range(nrof_batches_per_epoch):
        start_index = i * args.batch_size
        end_index = min((i + 1) * args.batch_size, nrof_images)
        paths_batch = paths[start_index:end_index]
        images = facenet.load_data(paths_batch, False, False, args.image_size)
        feed_dict = {images_placeholder: images, phase_train_placeholder: False}
        emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)

    classifier_filename_exp = os.path.expanduser(args.classifier_filename)

    if (args_mode == 'TRAIN'):
        # Train classifier
        print('Training classifier+++++++++++++++++++++++++', args.classifier)
        if args.classifier == 'LinearSvm':
            # clf = SVC(C=1, kernel='linear', probability=True)
            model = SVC(kernel='linear', probability=True)
        elif args.classifier == 'GridSearchSvm':
            print("""
                            Warning: In our experiences, using a grid search over SVM hyper-parameters only
                            gives marginally better performance than a linear SVM with C=1 and
                            is not worth the extra computations of performing a grid search.
                            """)
            param_grid = [
                {'C': [1, 10, 100, 1000],
                 'kernel': ['linear']},
                {'C': [1, 10, 100, 1000],
                 'gamma': [0.001, 0.0001],
                 'kernel': ['rbf']}
            ]
            model = GridSearchCV(SVC(C=1, probability=True), param_grid, cv=5)
        elif args.classifier == 'GMM':  # Doesn't work best
            model = GMM(n_components=nClasses)

        # ref:
        # http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#example-classification-plot-classifier-comparison-py
        elif args.classifier == 'RadialSvm':  # Radial Basis Function kernel
            # works better with C = 1 and gamma = 2
            model = SVC(C=1, kernel='rbf', probability=True, gamma=2)
        elif args.classifier == 'DecisionTree':  # Doesn't work best
            model = DecisionTreeClassifier(max_depth=20)
        elif args.classifier == 'GaussianNB':
            model = GaussianNB()

        # ref: https://jessesw.com/Deep-Learning/
        elif args.classifier == 'DBN':
            from nolearn.dbn import DBN
            model = DBN([embeddings.shape[1], 500, labelsNum[-1:][0] + 1],  # i/p nodes, hidden nodes, o/p nodes
                        learn_rates=0.3,
                        # Smaller steps mean a possibly more accurate result, but the
                        # training will take longer
                        learn_rate_decays=0.9,
                        # a factor the initial learning rate will be multiplied by
                        # after each iteration of the training
                        epochs=300,  # no of iternation
                        # dropouts = 0.25, # Express the percentage of nodes that
                        # will be randomly dropped as a decimal.
                        verbose=1)
        elif args.classifier == 'KNN':
            model = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                                         metric_params=None, n_jobs=1, n_neighbors=5, p=2,
                                         weights='distance')

        model.fit(emb_array, labels)

        # Create a list of class names
        class_names = [cls.name.replace('_', ' ') for cls in dataset]

        # Saving classifier model
        with open(classifier_filename_exp, 'wb') as outfile:
            pickle.dump((model, class_names), outfile)
        print('Saved classifier model to file "%s"' % classifier_filename_exp)

    elif (args_mode == 'CLASSIFY'):
        # Classify images
        print('Testing classifier~~~~~~~~~~~~~~~~~~~~~~~~')
        with open(classifier_filename_exp, 'rb') as infile:
            (model, class_names) = pickle.load(infile)
        predictions = np.zeros((nrof_images, len(class_names)))
        print('Loaded classifier model from file "%s"' % classifier_filename_exp)
        correctPrediction = 0
        inCorrectPrediction = 0
        sumConfidence = 0.0
        correctConfidence = 0.0
        inCorrectConfidence = 0.0
        '''
         batch_size =args.batch_size
        #batch_size = 1
        for i in range(nrof_batches_per_epoch):
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, nrof_images)
            starttime = time.time()
            mini_emb_array = emb_array[start_index:end_index, :]
            predictions[start_index:end_index, :] = model.predict_proba(mini_emb_array)
            print("start_index:{} end_index:{} time:{}".format(start_index, end_index, time.time() - starttime))

        '''
        predictions = model.predict_proba(emb_array)
        best_class_indices = np.argmax(predictions, axis=1)
        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
        results = {'name': [], 'bestname': [], 'probabilities': []}
        for i in range(len(best_class_indices)):
            # print(len(class_names))
            # print(i,len(labels),labels[i])
            # print(i,len(best_class_indices),best_class_indices[i])
            print('%4d  %s:%s: %.3f' % (
                i, class_labels[i], class_names[best_class_indices[i]], best_class_probabilities[i]))
            results['name'].append(class_labels[i])
            results['bestname'].append(class_names[best_class_indices[i]])
            results['probabilities'].append(best_class_probabilities[i])
            sumConfidence += best_class_probabilities[i]
            if (class_labels[i] == class_names[best_class_indices[i]]):
                correctPrediction += 1
                correctConfidence += best_class_probabilities[i]
            else:
                inCorrectPrediction += 1
                inCorrectConfidence += best_class_probabilities[i]

        # accuracy = np.mean(np.equal(best_class_indices, labels))
        accuracy = float(correctPrediction) / (correctPrediction + inCorrectPrediction)
        Avg_Confidence = float(sumConfidence) / (correctPrediction + inCorrectPrediction)
        Avg_correctConfidence = float(correctConfidence / correctPrediction)
        Avg_inCorrectConfidence = float(inCorrectConfidence / inCorrectPrediction)
        results['name'].append('Accuracy:')
        results['bestname'].append('Accuracy:')
        results['probabilities'].append(accuracy)
        dataname = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
        data_frame = pd.DataFrame(
            data={'name': results['name'], 'bestname': results['bestname'], 'probabilities': results['probabilities']})
        data_frame.to_csv(args.data_dir + '/results_' + dataname + '.csv')

        print("Correct Prediction :" + str(correctPrediction))
        print("In-correct Prediction: " + str(inCorrectPrediction))
        print('Accuracy: %.3f' % accuracy)
        print("Avg Confidence: " + str(Avg_Confidence))
        print("Avg CorrectConfidence: " + str(Avg_correctConfidence))
        print("Avg inCorrectConfidence: " + str(Avg_inCorrectConfidence))


def main(args):
    with tf.Graph().as_default():
        os.environ['CUDA_VISIBLE_DEVICES'] = '3'
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8,allow_growth = True)

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            # with tf.Session() as sess:
            np.random.seed(seed=args.seed)
            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(args.model)
            if args.use_split_dataset:
                dataset_tmp = facenet.get_dataset(args.data_dir)
                train_set, test_set = split_dataset(dataset_tmp, args.min_nrof_images_per_class,
                                                    args.nrof_train_images_per_class)
                classifier(args, 'TRAIN', train_set, sess)
                classifier(args, 'CLASSIFY', test_set, sess)

            else:
                dataset = facenet.get_dataset(args.data_dir)
                classifier(args, args.mode, dataset, sess)


def split_dataset(dataset, min_nrof_images_per_class, nrof_train_images_per_class):
    train_set = []
    test_set = []
    for cls in dataset:
        paths = cls.image_paths
        # Remove classes with less than min_nrof_images_per_class
        if len(paths) >= min_nrof_images_per_class:
            np.random.shuffle(paths)
            train_set.append(facenet.ImageClass(cls.name, paths[:nrof_train_images_per_class]))
            test_set.append(facenet.ImageClass(cls.name, paths[nrof_train_images_per_class:]))
    return train_set, test_set


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('mode', type=str, choices=['TRAIN', 'CLASSIFY'],
                        help='Indicates if a new classifier should be trained or a classification ' +
                             'model should be used for classification', default='CLASSIFY')
    parser.add_argument('--data_dir', type=str,
                        help='Path to the data directory containing aligned LFW face patches.')
    parser.add_argument(
        '--classifier',
        type=str,
        choices=[
            'LinearSvm',
            'KNN',
            'GMM',
            'RadialSvm',
            'DecisionTree'],
        help='The type of classifier to use.',
        default='KNN')
    parser.add_argument('--model', type=str,
                        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('--classifier_filename',
                        help='Classifier model file name as a pickle (.pkl) file. ' +
                             'For training this is the output and for classification this is an input.')
    parser.add_argument('--use_split_dataset',
                        help='Indicates that the dataset specified by data_dir should be split into a training and test set. ' +
                             'Otherwise a separate test set can be specified using the test_data_dir option.',
                        action='store_true')
    parser.add_argument('--test_data_dir', type=str,
                        help='Path to the test data directory containing aligned images used for testing.')
    parser.add_argument('--batch_size', type=int,
                        help='Number of images to process in a batch.', default=100)
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--seed', type=int,
                        help='Random seed.', default=666)
    parser.add_argument('--min_nrof_images_per_class', type=int,
                        help='Only include classes with at least this number of images in the dataset', default=45)
    parser.add_argument('--nrof_train_images_per_class', type=int,
                        help='Use this number of images from each class for training and the rest for testing',
                        default=30)

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
