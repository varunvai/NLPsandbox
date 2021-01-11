import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch
import numpy as np

class seq_classifier(nn.Module):

    def __init__(self, embed_size, h_size):
        super(seq_classifier, self).__init__()

        # Load glove embeddings
        fname ='glove.6B/glove.6B.'+str(embed_size) +'d.txt'
        self.glove_dict = {}
        with open(fname, encoding='utf-8') as f:
            for line in f:
                chars = line.split()
                self.glove_dict[chars[0]] = np.asarray(chars[1:], dtype=np.float32)
        self.embed_dim = len(self.glove_dict['the'])
        self.glove_dict['<PAD>'] = np.zeros(self.embed_dim)
        self.glove_dict['<DNE>'] = np.zeros(self.embed_dim)
        self.tokens = self.glove_dict.keys()

        self.recurrent_layer = nn.LSTM(input_size=embed_size,
                                       hidden_size=h_size,
                                       num_layers=1,
                                       bias=True,
                                       batch_first=True,
                                       bidirectional=True)
        self.linear = nn.Linear(h_size, 1, bias=True)
        self.linear_back = nn.Linear(h_size, 1, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, sents, lengths):
        # Pull embeddings from glove dictionary and pack sequence for RNN
        embeds = torch.Tensor([[self.glove_dict[token] if token in self.tokens else self.glove_dict['<DNE>']
                                       for token in sent]
                                      for sent in sents])
        packed_seq = pack_padded_sequence(embeds, lengths, batch_first=True)

        # Send packed sequence through recurrent layer.
        hts, (hn, cn) = self.recurrent_layer(packed_seq)
        return self.sigmoid(self.linear(hn[0]) + self.linear_back(hn[1]))[:,0]


class seq_classifier2(nn.Module):

    def __init__(self, embed_size, h_size, vocab, pretrained_embeds=True):
        super(seq_classifier2, self).__init__()

        # Load glove embeddings
        self._word2idx = vocab.word2idx
        self._idx2word = vocab.idx2word

        if pretrained_embeds:
            embed_weights = np.random.randn(len(vocab.word2idx), embed_size)
            fname ='glove.6B/glove.6B.'+str(embed_size) +'d.txt'
            with open(fname, encoding='utf-8') as f:
                for line in f:
                    chars = line.split()
                    word = chars[0]
                    if word in self._word2idx.keys():
                        idx = self._word2idx[word]
                        embed_weights[idx] = np.asarray(chars[1:], dtype=np.float32)
            self.embeddings = nn.Embedding(len(vocab.word2idx), embed_size)
            self.embeddings.weight.data.copy_(torch.from_numpy(embed_weights))
#            self.embeddings.weight.requires_grad = False
        else:
            self.embeddings = nn.Embedding(len(vocab.word2idx), embed_size)

        self.recurrent_layer = nn.LSTM(input_size=embed_size,
                                       hidden_size=h_size,
                                       num_layers=1,
                                       bias=True,
                                       batch_first=True,
                                       bidirectional=True)
        self.linear = nn.Linear(h_size, 1, bias=True)
        self.linear_back = nn.Linear(h_size, 1, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, word_idxs, lengths):
        # Pull embeddings from glove dictionary and pack sequence for RNN
        embeds = self.embeddings(word_idxs)
        packed_seq = pack_padded_sequence(embeds, lengths, batch_first=True)

        # Send packed sequence through recurrent layer
        hts, (hn, cn) = self.recurrent_layer(packed_seq)
        return self.sigmoid(self.linear(hn[0]) + self.linear_back(hn[1]))[:,0]