import torchtext
from torchtext.datasets import UDPOS

# Load the dataset splits
train_iter, valid_iter, test_iter = UDPOS(split=('train', 'valid', 'test'))

# Inspect a sample
for tokens, pos_tags, _ in train_iter:
    print("Tokens:", tokens)
    print("POS Tags:", pos_tags)
    break
