import torch.nn as nn
import torch
import pandas as pd
import numpy as np
from nltk.tokenize import TweetTokenizer
import tqdm

def batchgen(dataset, batchsize):
    n_batches = int(len(dataset)/batchsize)
    # Some datasets are sorted by their target. The following shuffle
    # will remove any ordering allowing random samples to be passed to
    # the training routine.
    idx_shuffle = np.arange(len(dataset))
    np.random.shuffle(idx_shuffle)
    tknzr = TweetTokenizer()
    for i in range(n_batches):
        " Blog authorship corpus "
        #tokens = [tknzr.tokenize(dataset.iloc[idx_shuffle[i*batchsize+j]]['text'].lower()) for j in range(batchsize)]
        #tgt = [int(dataset.iloc[idx_shuffle[i*batchsize+j]]['gender']=='female') for j in range(batchsize)]
        " Tripadvisor hotel reviews "
        #tokens = [tknzr.tokenize(dataset.iloc[idx_shuffle[i*batchsize+j]]['Review'].lower()) for j in range(batchsize)]
        #tgt = [int(int(dataset.iloc[idx_shuffle[i*batchsize+j]]['Rating']))>=4 for j in range(batchsize)]
        " IMDB reviews "
        text = [tknzr.tokenize(dataset.iloc[idx_shuffle[i*batchsize+j]]['review'].lower()) for j in range(batchsize)]
        tgt = [int(dataset.iloc[idx_shuffle[i*batchsize+j]]['sentiment']=='positive') for j in range(batchsize)]
        text, lengths, tgt = sort_and_pad(text, tgt)
        yield text, torch.Tensor(lengths), torch.FloatTensor(tgt)

def sort_and_pad(sents, tgt):
    """
    This pads all text in a batch to the length of the longest sequence.
    Any zero-length sequences are removed. The batch is also reordered
    to have descending sequence length (requirement for pytorch v1.0 packed sequences).
    :param sents:
    :param tgt:
    :return:
    """
    lengths = [len(sent) for sent in sents]
    idx_sort = np.flip(np.argsort(lengths))
    pad_length = max(lengths)
    sents_sorted = [sents[i]+['<PAD>']*(pad_length-len(sents[i])) for i in idx_sort if lengths[i]!=0]
    tgt_sorted = [tgt[i] for i in idx_sort if lengths[i]!=0]
    lengths_sorted = [lengths[i] for i in idx_sort if lengths[i]!=0]
    return sents_sorted, lengths_sorted, tgt_sorted

class vocab:
    def __init__(self, dataset):
        tknzr = TweetTokenizer()
        self.word2idx = {}
        idx = 1
        for text in tqdm.tqdm(dataset['review']):
            text_tokenized = tknzr.tokenize(text.lower())
            for word in text_tokenized:
                if word not in self.word2idx.keys():
                    self.word2idx[word] = idx
                    idx += 1
        self.word2idx['<PAD>'] = 0
        self.idx2word = {self.word2idx[key]: key for key in self.word2idx.keys()}

    def batch2idx(self, batch):
        return torch.LongTensor([[self.word2idx[word] for word in text] for text in batch])

if __name__ =='__main__':
    #data = pd.read_csv('blogtext.csv')
    data = pd.read_csv('IMDB Dataset.csv')
    Vocab = vocab(data)
    #batches = batchgen(data, 100)
