import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.ticker import NullFormatter
from sklearn.manifold import TSNE
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
import tensorflow as tf
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

def visualisation(final_result, summary_writer, sess, saver, log_dir, step):
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    META_FIEL = "face_meta.tsv"
    path_for_metadata = os.path.join(log_dir, META_FIEL)

    y = tf.Variable(final_result, name="embedding")
    config = projector.ProjectorConfig()

    embedding = config.embeddings.add()

    embedding.tensor_name = y.name

    # Specify where you find the metadata

    embedding.metadata_path = path_for_metadata

    # Say that you want to visualise the embeddings

    projector.visualize_embeddings(summary_writer, config)
    saver.save(sess, log_dir, step)

def log_evaluate(logdir, acc_mean, acd_std,val, val_std, far,auc,eer,epoch):

    log_file_path = "{}/accuracies.txt".format(logdir)
    if os.path.exists(log_file_path):
        print("{}/accuracies.txt already exists. Skipping processing.".format(logdir))
        file = open(log_file_path, "a")
    else:
        file = open(log_file_path, "w")
    file.write('*******************%d*****************\n' % (epoch))
    file.write('Accuracy: %1.3f+-%1.3f\n' % (acc_mean, acd_std))
    file.write('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f\n' % (val, val_std, far))
    file.write('Area Under Curve (AUC): %1.3f\n' % auc)
    file.write('Equal Error Rate (EER): %1.3f\n' % eer)

    file.close()