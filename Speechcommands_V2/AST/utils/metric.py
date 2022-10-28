import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import metrics

def d_prime(auc):
    standard_normal = stats.norm()
    d_prime = standard_normal.ppf(auc) * np.sqrt(2.0)
    return d_prime

def calculate_statistics(predictions, labels):
    """Calculate statistics including mAP, AUC, etc.

    Args:
    predictions: 2d array, (samples_num, classes_num)
    labels: 2d array, (samples_num, classes_num)        

    Returns:
    statistics: list of statistic of each class.
    """
    
    classes_num = len(labels[0])
    statistics = []
    
    
    # Accuracy, only used for single-label classification such as esc-50, not for multiple label one such as AudioSet
    acc = metrics.accuracy_score(np.argmax(labels, 1), np.argmax(predictions, 1))
    
    # Class-wise statistics
    for k in range(classes_num):

        # Average precision
        avg_precision = metrics.average_precision_score(
            labels[:, k], predictions[:, k], average=None)

        # AUC
        auc = metrics.roc_auc_score(labels[:, k], predictions[:, k], average=None)

        # Precisions, recalls
        (precisions, recalls, thresholds) = metrics.precision_recall_curve(
            labels[:, k], predictions[:, k])

        # FPR, TPR
        (fpr, tpr, thresholds) = metrics.roc_curve(labels[:, k], predictions[:, k])

        save_every_steps = 1000     # Sample statistics to reduce size
        dict = {'precisions': precisions[0::save_every_steps],
                'recalls': recalls[0::save_every_steps],
                'AP': avg_precision,
                'fpr': fpr[0::save_every_steps],
                'fnr': 1. - tpr[0::save_every_steps],
                'auc': auc,
                # note acc is not class-wise, this is just to keep consistent with other metrics
                'acc': acc
                }
        statistics.append(dict)

    return statistics

def calculate_metric(stats, cum_stats):
                
    mAP = np.mean([stat['AP'] for stat in stats])
    mAUC = np.mean([stat['auc'] for stat in stats])
    acc = stats[0]['acc']

    middle_ps = [stat['precisions'][int(len(stat['precisions'])/2)] for stat in stats]
    middle_rs = [stat['recalls'][int(len(stat['recalls'])/2)] for stat in stats]
    average_precision = np.mean(middle_ps)
    average_recall = np.mean(middle_rs)

    cum_mAP = np.mean([stat['AP'] for stat in cum_stats])
    cum_mAUC = np.mean([stat['auc'] for stat in cum_stats])
    cum_acc = cum_stats[0]['acc']

    return acc, mAP, mAUC, d_prime(mAUC), average_precision, average_recall, cum_mAP, cum_mAUC, cum_acc
    

def draw_histogram(scores, labels):
    positive = []
    negative = []
    for i in range(len(labels)):
        if int(labels[i]) == 1:
            positive.append(scores[i])
        else:
            negative.append(scores[i])
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.ylim([0, 500])
    ax.hist(positive, label="target", bins=200, color="blue", alpha=0.5)
    ax.hist(negative, label="nontarget", bins=200, color="red", alpha=0.5)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    plt.ylabel("#trial")
    plt.legend(loc="best")

    if os.path.exists('histogram.png'):
        os.remove('histogram.png')    

    plt.savefig("histogram.png", dpi=400, bbox_inches="tight")
    plt.close(fig)
    img = Image.open("histogram.png")
    
    return img

def draw_heatmap(title, x_axis, y_axis, featmap):
    plt.matshow(featmap, cmap=plt.get_cmap('GnBu'), aspect='auto')
    plt.colorbar(shrink=0.8, aspect=10)
    plt.clim(np.min(featmap), np.max(featmap))
    plt.title(title, fontsize=20)
    plt.xlabel(x_axis, fontsize=14)
    plt.ylabel(y_axis, fontsize=14)
    plt.savefig("heatmap.png", dpi=400, bbox_inches="tight")
    plt.close()
    img = Image.open("heatmap.png")
    os.remove('heatmap.png')

    return img