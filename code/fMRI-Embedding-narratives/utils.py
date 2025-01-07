import torch


def accuracy(predict, target, topk=(1, )):       
    """ Computes the Accuracy over the Top-K Predictions """
    # output.shape (batch_size, num_classes), target.shape (batch_size, )

    with torch.no_grad():
        maxk = max(topk)
        batch_size = predict.size(0)

        _, pred = predict.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


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