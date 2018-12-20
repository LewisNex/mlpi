from numpy import linspace
from sklearn.metrics import precision_recall_curve, confusion_matrix, roc_curve, roc_auc_score

import numpy as np
import itertools

import matplotlib.pyplot as plt
from matplotlib.cm import Blues_r as CMAP

def plot_roc(axis, y_true, y_scores):
    cmap = CMAP(linspace(0,1,1))
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    axis.plot(fpr, tpr, lw=2, label=f"ROC AUC: {auc:.2f}", color=cmap[0])
    axis.plot([0,1], [0,1], "k--")
    axis.axis([0, 1, 0, 1])
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")

def plot_prec_recall(axis, y_true, y_scores):
    cmap = CMAP(linspace(0,1,2))
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    axis.plot(thresholds, precisions[:-1], "--", label="Precision", color=cmap[0])
    axis.plot(thresholds, recalls[:-1], "--", label="Recall", color=cmap[1])
    axis.set_xlabel("Threshold")
    
def plot_confusion_matrix(axis, cm, classes,
                          normalize=False,
                          title='Confusion matrix',):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    axis.imshow(cm, interpolation='nearest', cmap=CMAP)
    tick_marks = np.arange(len(classes))
    axis.set_xticks(tick_marks)
    axis.set_yticks(tick_marks)
    axis.set_xticklabels(classes)
    axis.set_yticklabels(classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        axis.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] < thresh else "black")

    axis.set_ylabel('True label')
    axis.set_xlabel('Predicted label')
    
def plot_evaluations(y_scores, y_pred, y_true):
    # Set up the axes
    fig, (cm_axis, pr_axis, roc_axis) = plt.subplots(1, 3, figsize=(10, 3), gridspec_kw={'width_ratios':[2,5,5], "wspace":0.3})
    plot_prec_recall(pr_axis, y_true, y_scores)
    # Make ROC curve
    plot_roc(roc_axis, y_true, y_scores)
    #Make confusion matrix
    plot_confusion_matrix(cm_axis, confusion_matrix(y_true, y_pred), classes="gW", normalize=True)
    plt.legend()
    
    
def ceil(a, b=1):
    return int(((a - 1) // b) + 1)

def plot_hists(data, features, axes=None, **kwargs):
    if axes is None:
        _, axes = plt.subplots(2, ceil(len(features), 2), figsize=(len(features) * 5/2, 8))
    data[features].hist(ax=axes.flatten()[:len(features)], **kwargs)
    return axes

def plot_var_hists(data, features, bounds, grouping="pt", save_path=None):
    xbounds = {feature:(data[feature].min(), data[feature].max()) for feature in features}
    if grouping is "pt":
        for lb, ub in zip(bounds[:-1], bounds[1:]):
            subdata = data[data["pt"].between(lb, ub)]
            N = subdata.shape[0]
            fig, axes = plt.subplots(2, ceil(len(features), 2), figsize=(len(features) * 5/2, 8))
            _ = subdata.groupby("type").hist(ax=axes.flatten()[:len(features)], column=features, alpha=0.3, grid=False, density=True, bins="auto", ylabelsize=0)
            for axis in axes.flatten():
                feature = axis.get_title()
                if feature is not "":
                    axis.set_xlim(xbounds[feature])
            if save_path:
                plt.savefig(save_path + f"type_binned_{lb:d}_{ub:d}_N{N:d}")
                plt.clf()
        if save_path:
            return True
    else:
        for jet_type in [0, 1]:
            type_split_data = data[data["type"] == jet_type][features + ["pt"]].assign(bound=-1)
            print(type_split_data.columns)
            for bound_index, (lb, ub) in enumerate(zip(bounds[:-1], bounds[1:])):
                mask = type_split_data["pt"].between(lb, ub)
                type_split_data["bound"][mask] = bound_index
            N = type_split_data.shape[0]
            fig, axes = plt.subplots(2, ceil(len(features), 2), figsize=(len(features) * 5/2, 8))
            _ = type_split_data.groupby("bound").hist(ax=axes.flatten()[:len(features)], column=features, alpha=0.3, grid=False, density=True, bins="auto", ylabelsize=0)
            jet_type_symbol = "W" if jet_type == 1 else "g"
            for axis in axes.flatten():
                feature = axis.get_title()
                if feature is not "":
                    axis.set_xlim(xbounds[feature])
            if save_path:
                plt.savefig(save_path + f"pt_binned_{jet_type_symbol}_N{N:d}")
                plt.clf()
        if save_path:
            return True
                
        