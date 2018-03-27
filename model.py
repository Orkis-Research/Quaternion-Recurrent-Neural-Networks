import torch.nn                                                     as nn
from   torch.autograd import Variable
from   complex_models import CRN, MultiLayerCRN, ConvolutionalCRN
from   complex_models import BidirectionalCRN                       as BiCRN
from   complex_models import BidirectionalConvolutionalCRN          as ConvBiCRN
from   complex_models import MultiLayerConvolutionalCRN             as MultiLayerConvCRN
from   real_conv_rnn  import MultiLayerConvolutionalRNN             as MultiLayerConvRNN
from   real_conv_rnn  import ConvolutionalRNN                       as ConvRNN
from   real_conv_rnn  import BidirectionalConvolutionalRNN          as ConvBiRNN
from   torch.nn.init  import orthogonal


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, in_channels=None, out_channels=None, kernel_size=None,
                 rec_kernel_size=None, stride=None, dilation=None, padding=None, dropout=0.5, dropout_type='regular',
                 bias=True, bidirectional=False, tie_weights=False, seed=1337, weight_init='unitary',
                 w_mod_init='orthogonal', modulation_activation='softmax'):
        super(RNNModel, self).__init__()
        self.operation = 'convolution' if nhid is None else 'linear'
        self.bidirectional = bidirectional
        if self.operation == 'convolution':
            arguments = {
                'in_channels'   :  in_channels,  'out_channels'    :  out_channels,
                'kernel_size'   :  kernel_size,  'rec_kernel_size' :  rec_kernel_size,
                'stride'        :  stride,       'dilation'        :  dilation,
                'padding'       :  padding,      'dropout'         :  dropout,
                'dropout_type'  :  dropout_type, 'seed'            :  seed,
                'rnn_type'      :  rnn_type,     'bias'            :  bias
            }
        else:
            arguments = {
                'input_size'    :  ninp,         'hidden_size'     :  nhid,
                'dropout'       :  dropout,      'dropout_type'    :  dropout_type,
                'seed'          :  seed,         'rnn_type'        :  rnn_type,
                'bias'          :  bias
            }


        # I - Embedding part 
        if self.operation == 'linear':
            self.drop = nn.Dropout(dropout)
        else:
            self.drop = nn.Dropout2d(dropout)

        if rnn_type not in {'CRU', 'CGRU', 'CLSTM'}:
            self.encoder = nn.Embedding(ntoken, ninp)
        else:
            self.encoder = nn.Embedding(ntoken, 2 * ninp)

        # II - RNN part
        # II.1 {'LSTM', 'GRU', 'CRU', 'CGRU', 'CLSTM'}
        if rnn_type in {'LSTM', 'GRU', 'CRU', 'CGRU', 'CLSTM'}:

            # II.1.a
            if   rnn_type in {'LSTM', 'GRU'} and self.operation == 'linear':
                self.rnn = getattr(nn, rnn_type)(
                    ninp, nhid, nlayers, dropout=dropout,
                    bidirectional=self.bidirectional, bias=bias)

            # II.1.b Real-valued models
            elif rnn_type in {'GRU', 'LSTM'} and self.operation == 'convolution':
                if ((not self.bidirectional) and nlayers == 1):
                    RNN = ConvRNN
                elif (   self.bidirectional  and nlayers == 1):
                    RNN = ConvBiRNN
                elif nlayers > 1:
                    RNN = MultiLayerConvRNN
                    out_channels_list = (out_channels,) * nlayers
                    del arguments['out_channels']
                    arguments.update({'out_channels_list' : out_channels_list})
                    arguments.update({'bidirectional'     : self.bidirectional})
                del arguments['dropout_type']
                arguments.update({'weight_init' : weight_init})
                self.rnn = RNN(**arguments)
            
            # II.1.c Complex-valued models
            elif rnn_type in {'CRU', 'CGRU', 'CLSTM'}:
                if ((not self.bidirectional) and nlayers == 1):
                    RNN = CRN   if self.operation == 'linear' else ConvolutionalCRN
                elif (self.bidirectional and nlayers == 1):
                    RNN = BiCRN if self.operation == 'linear' else ConvBiCRN
                elif nlayers > 1:
                    if self.operation == 'convolution':
                        RNN = MultiLayerConvCRN
                        out_channels_list = (out_channels,) * nlayers
                        del arguments['out_channels']
                        arguments.update({'out_channels_list' : out_channels_list})
                        arguments.update({'bidirectional'     : self.bidirectional})
                        del arguments['dropout_type']
                    else:
                        layer_sizes = (nhid,) * nlayers
                        RNN = MultiLayerCRN
                        del arguments['hidden_size']
                        arguments.update({'layer_sizes'       : layer_sizes})
                        arguments.update({'bidirectional'     : self.bidirectional})
                arguments.update({'weight_init'           : weight_init,
                                  'w_mod_init'            : w_mod_init,
                                  'modulation_activation' : modulation_activation})
                self.rnn = RNN(**arguments)

        # II.2 {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'CRU', 'CGRU', 'CLSTM', RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        
        
        # III - Classification part
        # III.1 Real-Valued Models
        if rnn_type not in {'CRU', 'CGRU', 'CLSTM'}: #which means if the rnn is not complex
            nb_directions = 2 if self.bidirectional else 1
            if   self.operation == 'linear':
                self.decoder = nn.Linear(    nhid * nb_directions, ntoken)
            elif self.operation == 'convolution':
                self.decoder = nn.Linear(    ninp * out_channels * nb_directions, ntoken)
        # III.2 Complex-Valued Models
        else:
            nb_directions = 2 if self.bidirectional else 1
            if self.operation == 'linear':
                self.decoder = nn.Linear(2 * nhid * nb_directions, ntoken)
            else:
                self.decoder = nn.Linear(2 * ninp * out_channels * nb_directions, ntoken) # I am asuming that padding is of type 'same'

        
        # IV - Checking if weights are tied and Initialization.
        
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

        if rnn_type not in {'CRU', 'CGRU', 'CLSTM'}:
            if self.operation == 'linear': # this has to be changed. Even later we would want to initialize grus and lstms in our way.
                self.init_weights()

        self.rnn_type                =    rnn_type
        self.nhid                    =    nhid
        self.ninp                    =    ninp
        self.nlayers                 =    nlayers
        self.in_channels             =    in_channels
        self.out_channels            =    out_channels
        self.kernel_size             =    kernel_size
        self.rec_kernel_size         =    rec_kernel_size
        self.stride                  =    stride
        self.padding                 =    padding
        self.dilation                =    dilation
        self.dropout                 =    dropout,
        self.dropout_type            =    dropout_type
        self.bias                    =    bias
        self.tie_weights             =    tie_weights
        self.seed                    =    seed
        self.weight_init             =    weight_init
        self.w_mod_init              =    w_mod_init
        self.modulation_activation   =    modulation_activation

    def init_weights(self):
        orthogonal(self.encoder.weight)
        self.decoder.bias.data.fill_(0)
        orthogonal(self.decoder.weight)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input)) # emb size: (nb_timesteps, batch_size, embedding_dim)
        if self.operation == 'convolution':
            if self.rnn_type in {'CRU', 'CGRU', 'CLSTM'}:
                emb = emb.view(emb.size(0), emb.size(1), 2 * self.in_channels, self.ninp)
            else:
                emb = emb.view(emb.size(0), emb.size(1),     self.in_channels, self.ninp)
        output, hidden = self.rnn(emb, hidden)
        output  = self.drop(output) if self.operation == 'linear' else output
        decoded = self.decoder(output.view(output.size(0)*output.size(1), -1))
        # we will use as a criterion the nn.CrossEntropyLoss() which
        # applies in cascade the softmax and the negative log likelihood
        # See http://pytorch.org/docs/0.3.1/nn.html?highlight=cross%20entropy#torch.nn.functional.cross_entropy
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden


    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        nb_directions = 2 if self.bidirectional else 1
        if   self.rnn_type in {'LSTM', 'CLSTM'} and self.operation == 'linear':
            nhid = self.nhid if self.rnn_type == 'LSTM' else self.nhid * 2
            
            if self.rnn_type == 'LSTM':
                v =    (Variable(weight.new(self.nlayers * nb_directions, bsz, nhid).zero_()),
                        Variable(weight.new(self.nlayers * nb_directions, bsz, nhid).zero_()))
                return v

            elif self.rnn_type == 'CLSTM':
                v = (Variable(weight.new(self.nlayers, bsz, nhid).zero_()),
                     Variable(weight.new(self.nlayers, bsz, nhid).zero_()))
                if   nb_directions == 1 and self.nlayers == 1:
                    return v
                elif nb_directions > 1  and self.nlayers == 1:
                    u = (Variable(weight.new(self.nlayers, bsz, nhid).zero_()),
                         Variable(weight.new(self.nlayers, bsz, nhid).zero_()))
                    return (u, v)
                elif nb_directions == 1 and self.nlayers > 1:
                    return [(Variable(weight.new(bsz, nhid).zero_()),
                             Variable(weight.new(bsz, nhid).zero_())) \
                            for layer in range(self.nlayers)]
                elif nb_directions > 1 and self.nlayers > 1:
                    return [(
                        (Variable(weight.new(bsz, nhid).zero_()),
                         Variable(weight.new(bsz, nhid).zero_())),
                        (Variable(weight.new(bsz, nhid).zero_()),
                         Variable(weight.new(bsz, nhid).zero_()))
                    ) for layer in range(self.nlayers)]

        elif self.rnn_type in {'LSTM', 'CLSTM'} and self.operation == 'convolution':
            out_channels = self.out_channels if self.rnn_type == 'LSTM' else 2 * self.out_channels
            v   =  (Variable(weight.new(self.nlayers, bsz, out_channels, self.ninp).zero_()),
                    Variable(weight.new(self.nlayers, bsz, out_channels, self.ninp).zero_()))

            if   nb_directions == 1 and self.nlayers == 1:
                return v
            elif nb_directions == 1 and self.nlayers   > 1:
                return [(Variable(weight.new(bsz, out_channels, self.ninp).zero_()),
                         Variable(weight.new(bsz, out_channels, self.ninp).zero_())) \
                        for layer in range(self.nlayers)]

            elif nb_directions > 1:
                u = (Variable(weight.new(self.nlayers, bsz, out_channels, self.ninp).zero_()),
                     Variable(weight.new(self.nlayers, bsz, out_channels, self.ninp).zero_()))
                if self.nlayers == 1:
                    return (u, v)
                else:
                    return [(
                        (Variable(weight.new(bsz, out_channels, self.ninp).zero_()),
                         Variable(weight.new(bsz, out_channels, self.ninp).zero_())),
                        (Variable(weight.new(bsz, out_channels, self.ninp).zero_()),
                         Variable(weight.new(bsz, out_channels, self.ninp).zero_()))
                    ) for layer in range(self.nlayers)]


        elif self.rnn_type in {'CRU', 'CGRU'}:
            if not self.bidirectional:
                if self.operation   == 'convolution':
                    return Variable(weight.new(    self.nlayers, bsz, 2 * self.out_channels, self.ninp).zero_())
                elif self.operation == 'linear':
                    return Variable(weight.new(    self.nlayers, bsz, self.nhid * 2).zero_())
            else: # nb_directions here is going to be = 2
                if self.nlayers > 1:
                    if self.operation   == 'convolution':
                        return Variable(weight.new(self.nlayers, nb_directions, bsz, 2 * self.out_channels, self.ninp).zero_())
                    elif self.operation == 'linear':
                        return Variable(weight.new(self.nlayers, nb_directions, bsz, self.nhid * 2).zero_())
                elif self.nlayers == 1:
                    if self.operation   == 'convolution':
                        return Variable(weight.new(              nb_directions, bsz, 2 * self.out_channels, self.ninp).zero_())
                    elif self.operation == 'linear':
                        return Variable(weight.new(              nb_directions, bsz, self.nhid * 2).zero_())

        elif self.rnn_type == 'GRU':
            if self.operation  == 'convolution':
                if self.nlayers == 1:
                    return Variable(weight.new(              nb_directions,  bsz, self.out_channels, self.ninp).zero_())
                elif (not self.bidirectional) and (self.nlayers > 1):
                    return Variable(weight.new(self.nlayers,                 bsz, self.out_channels, self.ninp).zero_())
                elif      self.bidirectional  and  self.nlayers > 1:
                    return Variable(weight.new(self.nlayers, nb_directions,  bsz, self.out_channels, self.ninp).zero_())
            elif self.operation == 'linear':
                if self.nlayers == 1:
                    return Variable(weight.new(              nb_directions,  bsz, self.nhid).zero_())
                elif (not self.bidirectional) and (self.nlayers > 1):
                    return Variable(weight.new(self.nlayers,                 bsz, self.nhid).zero_())
                elif      self.bidirectional  and  self.nlayers > 1:
                    return Variable(weight.new(self.nlayers, nb_directions,  bsz, self.nhid).zero_())

        else:
            return Variable(weight.new(self.nlayers * nb_directions, bsz, self.nhid).zero_())
        # torch.Tensor.new constructs a new tensor of the same data type.
        # See http://pytorch.org/docs/0.3.0/tensors.html?highlight=new#torch.Tensor.new
