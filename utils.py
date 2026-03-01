import torch
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from easydict import EasyDict
from sklearn.metrics import roc_curve, auc, confusion_matrix
from scipy.optimize import brentq
from scipy.interpolate import interp1d


def cont_grad(x, rate=1):
    return rate * x + (1 - rate) * x.detach()

def find_best_threshold(y_trues, y_preds):
    '''
        This function is utilized to find the threshold corresponding to the best ACER
        Args:
            y_trues (list): the list of the ground-truth labels, which contains the int data
            y_preds (list): the list of the predicted results, which contains the float data
    '''
    print("Finding best threshold...")
    best_thre = 0.5
    best_metrics = None
    candidate_thres = list(np.unique(np.sort(y_preds)))
    for thre in candidate_thres:
        metrics = cal_metrics(y_trues, y_preds, threshold=thre)
        if best_metrics is None:
            best_metrics = metrics
            best_thre = thre
        elif metrics.ACER < best_metrics.ACER:
            best_metrics = metrics
            best_thre = thre
    print(f"Best threshold is {best_thre}")
    return best_thre, best_metrics


def cal_metrics(y_trues, y_preds, threshold=0.5):
    '''
        This function is utilized to calculate the performance of the methods
        Args:
            y_trues (list): the list of the ground-truth labels, which contains the int data
            y_preds (list): the list of the predicted results, which contains the float data
            threshold (float, optional):
                'best': calculate the best results
                'auto': calculate the results corresponding to the thresholds of EER
                float: calculate the results of the specific thresholds
    '''

    metrics = EasyDict()

    fpr, tpr, thresholds = roc_curve(y_trues, y_preds)
    metrics.AUC = auc(fpr, tpr)

    metrics.EER = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    metrics.Thre = float(interp1d(fpr, thresholds)(metrics.EER))

    if threshold == 'best':
        _, best_metrics = find_best_threshold(y_trues, y_preds)
        return best_metrics

    elif threshold == 'auto':
        threshold = metrics.Thre
        # print('Auto threshold is:',threshold)

    prediction = (np.array(y_preds) > threshold).astype(int)

    res = confusion_matrix(y_trues, prediction, labels=[0, 1])
    # confusion_matrix returns [[tn, fp], [fn, tp]] for labels=[0,1]
    tn, fp = res[0, :]
    fn, tp = res[1, :]
    metrics.ACC = (tp + tn) / len(y_trues)

    TP_rate = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    TN_rate = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # APCER = Attack Presentation Classification Error Rate = FP / (TN + FP)
    # BPCER = Bona Fide Presentation Classification Error Rate = FN / (FN + TP)
    metrics.APCER = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0
    metrics.BPCER = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    metrics.ACER = (metrics.APCER + metrics.BPCER) / 2

    return metrics


