"""Validate a face recognizer on the "Labeled Faces in the Wild" dataset (http://vis-www.cs.umass.edu/lfw/).
Embeddings are calculated using the pairs from http://vis-www.cs.umass.edu/lfw/pairs.txt and the ROC curve
is calculated and plotted. Both the model metagraph and the model parameters need to exist
in the same directory, and the metagraph should have the extension '.meta'.
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
import lfw
import os
import sys
import math
#import matplotlib.pyplot as plt
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate
from scipy.interpolate import interp1d
import pandas as pd
import matplotlib.pyplot as plt

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
def getAUC(fprs, tprs):
    sortedFprs, sortedTprs = zip(*sorted(zip(*(fprs, tprs))))
    sortedFprs = list(sortedFprs)
    sortedTprs = list(sortedTprs)
    if sortedFprs[-1] != 1.0:
        sortedFprs.append(1.0)
        sortedTprs.append(sortedTprs[-1])
    return np.trapz(sortedTprs, sortedFprs)

def getDistances(embeddings):
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    #Lx = np.sqrt(embeddings1.dot(embeddings1.T))
    Lx = np.sqrt(np.sum(np.square(embeddings1), 1))
    #np.dot(embeddings1.T, embeddings1)
    #Ly = np.sqrt(embeddings2.dot(embeddings2.T))
    Ly = np.sqrt(np.sum(np.square(embeddings2), 1))
    cos_angele = np.sum(embeddings1*embeddings2,1)/(Lx*Ly)
    return  dist,cos_angele

def plotOpenFaceROC(workDir, plotFolds=True, color=None):
    fs = []
    for i in range(10):
        rocData = pd.read_csv("{}/../comparisons/openface/l2-roc.fold-{}.csv".format(workDir, i))
        fs.append(interp1d(rocData['fpr'], rocData['tpr']))
        x = np.linspace(0, 1, 1000)
        if plotFolds:
            foldPlot, = plt.plot(x, fs[-1](x), color='grey', alpha=0.5)

        else:
            foldPlot = None

    fprs = []
    tprs = []
    for fpr in np.linspace(0, 1, 1000):
        tpr = 0.0
        for f in fs:
            v = f(fpr)
            if math.isnan(v):
                v = 0.0
            tpr += v
        tpr /= 10.0
        fprs.append(fpr)
        tprs.append(tpr)
    if color:
        meanPlot, = plt.plot(fprs, tprs, color=color)
    else:
        meanPlot, = plt.plot(fprs, tprs)
    AUC = getAUC(fprs, tprs)
    return foldPlot, meanPlot, AUC


def plotFacenetROC(workDir, plotFolds=True, color=None):
    fs = []
    for i in range(10):
        rocData = pd.read_csv("{}/../comparisons/facenet/l2-roc.fold-{}.csv".format(workDir, i))
        fs.append(interp1d(rocData['fpr'], rocData['tpr']))
        x = np.linspace(0, 1, 1000)
        if plotFolds:
            foldPlot, = plt.plot(x, fs[-1](x), color='grey', alpha=0.5)
        else:
            foldPlot = None

    fprs = []
    tprs = []
    for fpr in np.linspace(0, 1, 1000):
        tpr = 0.0
        for f in fs:
            v = f(fpr)
            if math.isnan(v):
                v = 0.0
            tpr += v
        tpr /= 10.0
        fprs.append(fpr)
        tprs.append(tpr)
    if color:
        meanPlot, = plt.plot(fprs, tprs, color=color)
    else:
        meanPlot, = plt.plot(fprs, tprs)
    AUC = getAUC(fprs, tprs)
    return foldPlot, meanPlot, AUC


def plotVerifyExp(workDir, tag):
    print("Plotting.")

    fig, ax = plt.subplots(1, 1)

    openbrData = pd.read_csv("../comparisons/openbr.v1.1.0.DET.csv")
    openbrData['Y'] = 1 - openbrData['Y']
    # brPlot = openbrData.plot(x='X', y='Y', legend=True, ax=ax)
    brPlot, = plt.plot(openbrData['X'], openbrData['Y'])
    brAUC = getAUC(openbrData['X'], openbrData['Y'])

    foldPlot, meanPlot, AUC = plotOpenFaceROC(workDir, color='k')
    netfoldPlot, netmeanPlot, netAUC = plotFacenetROC(workDir, color='r')

    humanData = pd.read_table(
        "../comparisons/kumar_human_crop.txt", header=None, sep=' ')
    humanPlot, = plt.plot(humanData[1], humanData[0])
    humanAUC = getAUC(humanData[1], humanData[0])

    deepfaceData = pd.read_table(
        "../comparisons/deepface_ensemble.txt", header=None, sep=' ')
    dfPlot, = plt.plot(deepfaceData[1], deepfaceData[0], '--',
                       alpha=0.75)
    deepfaceAUC = getAUC(deepfaceData[1], deepfaceData[0])

    # baiduData = pd.read_table(
    #     "comparisons/BaiduIDLFinal.TPFP", header=None, sep=' ')
    # bPlot, = plt.plot(baiduData[1], baiduData[0])
    # baiduAUC = getAUC(baiduData[1], baiduData[0])

    eigData = pd.read_table(
        "../comparisons/eigenfaces-original-roc.txt", header=None, sep=' ')
    eigPlot, = plt.plot(eigData[1], eigData[0])
    eigAUC = getAUC(eigData[1], eigData[0])

    ax.legend([humanPlot, dfPlot, brPlot, eigPlot,
               meanPlot, foldPlot, netmeanPlot, netfoldPlot],
              ['Human, Cropped [AUC={:.3f}]'.format(humanAUC),
               # 'Baidu [{:.3f}]'.format(baiduAUC),
               'DeepFace Ensemble [{:.3f}]'.format(deepfaceAUC),
               'OpenBR v1.1.0 [{:.3f}]'.format(brAUC),
               'Eigenfaces [{:.3f}]'.format(eigAUC),
               'OpenFace [{:.3f}]'.format(AUC),
               'OpenFace {} folds'.format(tag),
               'Facenet {} [{:.3f}]'.format(tag, netAUC),
               'Facenet {} folds'.format(tag)],
              loc='lower right')

    plt.plot([0, 1], color='k', linestyle=':')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    #plt.ylim(ymin=0,ymax=1)
    plt.xlim(xmin=0, xmax=1)

    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='k', linestyle='-', alpha=0.2)
    plt.minorticks_on()
    # fig.savefig(os.path.join(workDir, "roc.pdf"))
    fig.savefig(os.path.join(workDir, "roc.png"))

def plotExp(path,distance,cos,actual_issame):

    x = range(len(actual_issame))
    color_list = []
    for i in range(len(actual_issame)):
        if(actual_issame[i]==True):
            color_list.append('g')
        else:
            color_list.append('r')
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.scatter(x, distance, marker='o', c='', edgecolors=color_list)
    ax1.set_xlabel("num of pairs")
    ax1.set_ylabel('distance')
    ax2 = fig.add_subplot(2, 1, 2)

    ax2.scatter(x, cos, marker='o', c='', edgecolors=color_list)
    ax2.set_xlabel("num of pairs")
    ax2.set_ylabel('cos')
    fig.savefig(os.path.join(path, "distance.png"))
    fig.show()



def main(args):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Read the file containing the pairs used for testing
            pairs = lfw.read_pairs(os.path.expanduser(args.lfw_pairs))

            # Get the paths for the corresponding images
            paths, actual_issame = lfw.get_paths(os.path.expanduser(args.lfw_dir), pairs, args.lfw_file_ext)

            # Load the model
            facenet.load_model(args.model)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # image_size = images_placeholder.get_shape()[1]  # For some reason this doesn't work for frozen graphs
            image_size = args.image_size
            embedding_size = embeddings.get_shape()[1]

            # Run forward pass to calculate embeddings
            print('Runnning forward pass on LFW images')
            batch_size = args.lfw_batch_size
            nrof_images = len(paths)
            nrof_batches = int(math.ceil(1.0 * nrof_images / batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches):
                start_index = i * batch_size
                end_index = min((i + 1) * batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, image_size)
                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)

            distance, cos = getDistances(emb_array)
            # dataname = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
            data_frame = pd.DataFrame(
                data={'person1': paths[0::2], 'person2': paths[1::2],
                      'true': actual_issame,
                      'distance': distance, 'cos': cos})
            data_frame.to_csv(os.path.join(args.lfw_dir, "results_data_mini.csv"))
            print('lfw evaluate ........')
            tpr, fpr, accuracy, val, val_std, far = lfw.evaluate(emb_array,
                                                                     actual_issame, nrof_folds=args.lfw_nrof_folds)

            print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
            print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
            auc = metrics.auc(fpr, tpr)
            print('Area Under Curve (AUC): %1.3f' % auc)
            eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
            print('Equal Error Rate (EER): %1.3f' % eer)
            plotExp(args.lfw_dir,distance, cos, actual_issame)
            #plotVerifyExp(args.lfw_dir, 'roc')

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--lfw_dir', type=str,
                        default='D:\\study\\AI\\project\\data_proc\\mini_massvie\\valid_name_align_160',
                        help='Path to the data directory containing aligned LFW face patches.')
    parser.add_argument('--lfw_batch_size', type=int,
                        help='Number of images to process in a batch in the LFW test set.', default=10)


    parser.add_argument('--model', type=str,
                        default='D:\\study\\AI\\project\\facenet\\facevalid\models\\20180428-181544',
                        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)

    parser.add_argument('--lfw_pairs', type=str,
                        help='The file containing the pairs to use for validation.',
                        default='D:\\study\\AI\\project\\data_proc\\mini_massvie\\valid_name_align_160\\pairs.txt')
    parser.add_argument('--lfw_file_ext', type=str,
                        help='The file extension for the LFW dataset.', default='png', choices=['jpg', 'png'])
    parser.add_argument('--lfw_nrof_folds', type=int,
                        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
