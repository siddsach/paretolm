import torch.nn as nn
from torch.autograd import Variable
from adasoft import *

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5,tie_weights=False, adasoft=False, cutoff = [2000, 10000]):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)


        self.adasoft = adasoft
        if adasoft:
            self.decoder = AdaptiveSoftmax(nhid, [*cutoff, ntoken + 1])
        else:
            self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        if not self.adasoft:
            nn.init.xavier_normal(self.decoder.weight)
            self.decoder.bias.data.fill_(0)
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, target=None):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)

        if not self.adasoft:
            decoded = self.decoder(output
                    .view(output.size(0) * output.size(1), output.size(2)))
            decoded = decoded.view(output.size(0), output.size(1), decoded.size(1))

        else:
            if target is not None:
                self.decoder.set_target(target.data)
                decoded = self.decoder(output
                        .view(output.size(0) * output.size(1), output.size(2)))
            else:
                decoded = self.decoder.log_prob(output
                        .view(output.size(0) * output.size(1), output.size(2)))
                decoded = Variable(decoded)

        return decoded, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())

