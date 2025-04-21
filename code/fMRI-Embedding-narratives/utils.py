import torch
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC
from torcheval.metrics.functional import multiclass_accuracy

pos_tags = {
    'ADJ': 0,
    'ADP': 1,
    'ADV': 2,
    'AUX': 3,
    'CCONJ': 4,
    'DET': 5,
    'INTJ': 6,
    'NOUN': 7,
    'NUM': 8,
    'PART': 9,
    'PRON': 10,
    'PROPN': 11,
    'PUNCT': 12,
    'SCONJ': 13,
    'SYM': 14,
    'VERB': 15,
    'X': 16,
    'PAD' : 17,
    'START' : 18,
    'END' : 19
}


def accuracy(predict, target):       
    return multiclass_accuracy(predict, target, ignore_index=pos_tags['PAD'])


class AverageMeter(object):
    """ Computes and Storing the Avearge and Current Value """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

