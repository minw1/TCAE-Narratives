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

