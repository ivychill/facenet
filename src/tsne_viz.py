import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.ticker import NullFormatter
from sklearn.manifold import TSNE
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
import tensorflow as tf
from scipy.interpolate import interp1d
import math

def plot_tsne(embeddings,labels,logdir,step):
    #tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)

    #tsne = TSNE(n_components=2, init='pca', verbose=2)
    tsne = TSNE(n_components=2)
    embeddings = tsne.fit_transform(embeddings)
    num_classes = len(np.unique(labels))
    fig = plt.figure()
    #fig, ax = plt.subplots(1, 1)
    #subname = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    ax = fig.add_subplot(111)
    colors = cm.Spectral(np.linspace(0, 1, num_classes))
    classes = np.arange(num_classes)
    xx = embeddings[:, 0]
    yy = embeddings[:, 1]
    # plot the 2D data points
    labels = np.array(labels)
    cmarker = ['.','o',',','v','^','<','>','1','2','3','4','8','s','p','*','h','+','d']
    for i in range(num_classes):
        ax.scatter(xx[labels == i], yy[labels == i], color=colors[i], label=classes[i], s=10,
                   marker=cmarker[i % len(cmarker)])
        # index = list(np.where(labels == i))[0]
        # for k in range(len(index)):
        #     ccmarker = ['o',  '+']
        #     ccolors = ['r','b']
        #     ax.scatter(xx[index[k]], yy[index[k]], color=ccolors[0 if k < len(index)//2 else 1], label=classes[i], s=10,
        #            marker=cmarker[i % len(cmarker)])


    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')
    plt.legend(loc='best', scatterpoints=1, fontsize=5)
    #plt.savefig(os.path.join(logdir, "tnse_"+subname+".png"))
    plt.minorticks_on()
    plt.savefig(os.path.join(logdir, "tnse_"+str(step)+".png"),format='png')

def getAUC(fprs, tprs):
    sortedFprs, sortedTprs = zip(*sorted(zip(*(fprs, tprs))))
    sortedFprs = list(sortedFprs)
    sortedTprs = list(sortedTprs)
    if sortedFprs[-1] != 1.0:
        sortedFprs.append(1.0)
        sortedTprs.append(sortedTprs[-1])
    return np.trapz(sortedTprs, sortedFprs)

def plotFaceROC(logdir, epoch,tprs_folds, fprs_folds_,nrof_folds,color='k'):
    fig, ax = plt.subplots(1, 1)
    fs = []
    for i in range(nrof_folds):
        fs.append(interp1d(fprs_folds_[i], tprs_folds[i]))
        x = np.linspace(0, 1, 1000)
        foldPlot, = plt.plot(x, fs[-1](x), color='grey', alpha=0.5)

    fprs = []
    tprs = []
    for fpr in np.linspace(0, 1, 1000):
        tpr = 0.0
        for f in fs:
            v = f(fpr)
            if math.isnan(v):
                v = 0.0
            tpr += v
        tpr /= nrof_folds
        fprs.append(fpr)
        tprs.append(tpr)
    if color:
        meanPlot, = plt.plot(fprs, tprs, color=color)
    else:
        meanPlot, = plt.plot(fprs, tprs)
    AUC = getAUC(fprs, tprs)

    ax.legend([meanPlot, foldPlot],
              ['facenet {} [{:.3f}]'.format('roc', AUC),
               'facenet {} folds'.format('roc')],
              loc='lower right')

    plt.plot([0, 1], color='k', linestyle=':')

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    # plt.ylim(ymin=0,ymax=1)
    plt.xlim(xmin=0, xmax=1)

    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='k', linestyle='-', alpha=0.2)
    plt.minorticks_on()
    # fig.savefig(os.path.join(workDir, "roc.pdf"))

    fig.savefig(os.path.join(logdir, "roc_" + str(epoch) + ".png"), format='png')

def plotKS(logdir, tprs, fprs,thresholds,epoch):
    fig, ax = plt.subplots(1, 1)

    meanPlot, = plt.plot(thresholds, tprs, color='k')
    foldPlot, = plt.plot(thresholds, fprs, color='grey')

    ax.legend([meanPlot, foldPlot],
              ['facenet {} '.format('tpr'),
               'facenet {} '.format('fpr')],
              loc='lower right')



    plt.xlabel("thresholds")
    plt.ylabel("TPR/FPR")
    # plt.ylim(ymin=0,ymax=1)
    plt.xlim(xmin=0, xmax=4)

    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='k', linestyle='-', alpha=0.2)
    plt.minorticks_on()
    # fig.savefig(os.path.join(workDir, "roc.pdf"))

    fig.savefig(os.path.join(logdir, "KS_" + str(epoch) + ".png"), format='png')

def plotExp(logdir,distance,cos,actual_issame,epoch):

    x = range(len(actual_issame))
    color_list = []
    for i in range(len(actual_issame)):
        if actual_issame[i]:
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
    #fig.savefig(os.path.join(path, "distance.png"))
    fig.savefig(os.path.join(logdir, "distance_" + str(epoch) + ".png"), format='png')

def visualisation(final_result, summary_writer, sess, saver, log_dir, step):
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    META_FILE = "face_meta.tsv"
    path_for_metadata = os.path.join(log_dir, META_FILE)

    y = tf.Variable(final_result, name="embedding")
    config = projector.ProjectorConfig()

    embedding = config.embeddings.add()

    embedding.tensor_name = y.name

    # Specify where you find the metadata

    embedding.metadata_path = path_for_metadata

    # Say that you want to visualise the embeddings

    projector.visualize_embeddings(summary_writer, config)
    #saver.save(sess, log_dir, step)

def log_evaluate(logdir, acc_mean, acd_std, val, val_std, far,auc,eer,prefix,epoch):
    global saveno
    log_file_path = "{}/{}_accuracies.txt".format(logdir,prefix)
    if os.path.exists(log_file_path):
        print("{}/accuracies.txt already exists. Skipping processing.".format(logdir))
        file = open(log_file_path, "a")
    else:
        file = open(log_file_path, "w")

    file.write('*******************%s*****************\n' % (epoch))
    file.write('Accuracy: %1.3f+-%1.3f\n' % (acc_mean, acd_std))
    file.write('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f\n' % (val, val_std, far))
    file.write('Area Under Curve (AUC): %1.3f\n' % auc)
    file.write('Equal Error Rate (EER): %1.3f\n' % eer)

    file.close()