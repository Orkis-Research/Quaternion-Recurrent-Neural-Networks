import numpy as np
from numpy.random import RandomState
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import PackedSequence
from torch.nn import Module
from torch.nn._functions.rnn import Recurrent, variable_recurrent_factory
from quaternionops import *
from rnn import recurrent, QGRUCell, QLSTMCell, QRNNBaseCell
import math
import sys
import realops


class QRN(Module):
    r"""Applies a 1 layer complex recurrent unit (CRU) RNN to an input sequence.

    For each element in the input sequence, each layer computes the following
    function:

    Args:
        input_size: The number of expected complex features in the input x
        hidden_size: The number of complex features in the hidden state h
        bias: If ``False``, then the layer does not use bias weights b_ih and b_hh.
            Default: ``True``
        dropout: If non-zero, performs complex dropout on the outputs of each
            RNN layer except the last layer

    Inputs: input, h_0
        - **input** (seq_len, batch, 2 * input_size): complex tensor containing the features
          of the input sequence.
        - **h_0** (num_layers * num_directions, batch, 2 * hidden_size): complex tensor
          containing the initial hidden state for each element in the batch.

    Outputs: output, h_n
        - **output** (seq_len, batch, 2 * hidden_size * num_directions): tensor
          containing the output features h_t from the last layer of the RNN,
          for each t. If a :class:`torch.nn.utils.rnn.PackedSequence` has been
          given as the input, the output will also be a packed sequence.
        - **h_n** (num_layers * num_directions, batch, hidden_size): tensor
          containing the hidden state for t=seq_len

    """
    def __init__(self, input_size, hidden_size, rnn_type='QGRU',
                 bias=True, dropout=0, reverse=False,
                 init_criterion='glorot',
                 weight_init='unitary',
                 w_mod_init='orthogonal',
                 dropout_type='regular',
                 modulation_activation='softmax',
                 seed=None, **kwargs):
        super(QRN, self).__init__(**kwargs)
        self.input_size      =    input_size
        self.hidden_size     =    hidden_size
        self.bias            =    bias
        self.dropout         =    dropout
        self.dropout_type    =    dropout_type
        self.reverse         =    reverse
        self.init_criterion  =    init_criterion
        self.weight_init     =    weight_init
        self.seed            =    seed if seed is not None else 1337
        self.rng             =    RandomState(self.seed)
        self.winit           =    {'quaternion': quaternion_init,
                                   'unitary': unitary_init}[self.weight_init]
        if w_mod_init is not None:
            self.wmodinit    =    {'orthogonal': realops.independent_filters_init}[w_mod_init]
        if modulation_activation is not None:
            self.modact      =    modulation_activation

        self.rnn_type        =    rnn_type

        if self.rnn_type in {'QRNN'}:
            gate_size        =    self.hidden_size
        elif self.rnn_type in {'QGRU'}:
            gate_size        =    3 * self.hidden_size
        elif self.rnn_type in {'QLSTM'}:
            gate_size        =    4 * self.hidden_size

        # define parameters:
        self.w_ih_r       =    Parameter(torch.Tensor(self.input_size,  gate_size))
        self.w_ih_i       =    Parameter(torch.Tensor(self.input_size,  gate_size))
        self.w_ih_j       =    Parameter(torch.Tensor(self.input_size,  gate_size))
        self.w_ih_k       =    Parameter(torch.Tensor(self.input_size,  gate_size))
        self.w_hh_r       =    Parameter(torch.Tensor(self.hidden_size, gate_size))
        self.w_hh_i       =    Parameter(torch.Tensor(self.hidden_size, gate_size))
        self.w_hh_j       =    Parameter(torch.Tensor(self.hidden_size, gate_size))
        self.w_hh_k       =    Parameter(torch.Tensor(self.hidden_size, gate_size))

        if self.bias:
            self.b_ih        =    Parameter(torch.Tensor(gate_size * 4))
            self.b_hh        =    Parameter(torch.Tensor(gate_size * 4))
        else:
            self.register_parameter('bias_ih',        None)
            self.register_parameter('bias_hh',        None)

        self.reset_parameters()

    def reset_parameters(self):
        fargs = [self.winit, self.rng, self.init_criterion]
        affect_init(self.w_ih_r, self.w_ih_i, self.w_ih_j, self.w_ih_k, *fargs)
        affect_init(self.w_hh_r, self.w_hh_i, self.w_hh_j, self.w_hh_k, *fargs)

        if self.bias:
            self.b_ih.data.zero_()
            self.b_hh.data.zero_()

    def forward(self, input, hx):
        params = {
            'w_ih_r': self.w_ih_r,
            'w_ih_i': self.w_ih_i,
            'w_ih_j': self.w_ih_j,
            'w_ih_k': self.w_ih_k,
            'w_hh_r': self.w_hh_r,
            'w_hh_i': self.w_hh_i,
            'w_hh_j': self.w_hh_j,
            'w_hh_k': self.w_hh_k,
            'b_ih': self.b_ih,
            'b_hh': self.b_hh,
            'dropout': self.dropout,
            'dropout_type': self.dropout_type,
            'train': self.training,
            'rng': self.rng
        }

        cellname = {'QGRU': QGRUCell, 'QLSTM': QLSTMCell, 'QRNN': QRNNBaseCell}[self.rnn_type]
        return recurrent(cellname, input, hx, reverse=self.reverse, **params)


class BidirectionalQRN(Module):

    def __init__(self, input_size, hidden_size,
                 bias=True, dropout=0, dropout_type='regular',
                 rnn_type='QGRU', init_criterion='glorot',
                 weight_init='quaternion',
                 seed=None, **kwargs):

        super(BidirectionalCRN, self).__init__()
        self.rnn_type       = rnn_type
        self.forwardqrn   = QRN(input_size, hidden_size, rnn_type=self.rnn_type,
                                bias=bias, dropout=dropout, dropout_type=dropout_type,
                                reverse=False, init_criterion=init_criterion,
                                weight_init=weight_init,
                                seed=seed, **kwargs)
        self.backwardqrn  = QRN(input_size, hidden_size, rnn_type=self.rnn_type,
                                bias=bias, dropout=dropout, dropout_type=dropout_type,
                                reverse=True, init_criterion=init_criterion,
                                weight_init=weight_init,
                                seed=seed, **kwargs)

    def reset_parameters(self):
        self.forwardqrn.reset_parameters()
        self.backwardqrn.reset_parameters()

    def forward(self, input, hx):
        if   self.rnn_type in {'QLSTM'}:
            o1, lasth1 = self.forwardqrn (input, (hx[0][0], hx[0][1]))
            o2, lasth2 = self.backwardqrn(input, (hx[1][0], hx[1][1]))
        else:
            o1, lasth1 = self.forwardqrn (input, hx[0].unsqueeze(0))
            o2, lasth2 = self.backwardqrn(input, hx[1].unsqueeze(0))
        out   = torch.cat([o1, o2], dim=-1)
        hlast = torch.cat([o1[-1].unsqueeze(0), o2[-1].unsqueeze(0)], dim=0)
        return out, hlast


class MultiLayerQRN(Module):
    """heavily inspired from:
       https://github.com/pytorch/benchmark/blob/master/benchmarks/lstm_variants/container.py
    """
    def __init__(self, input_size, bidirectional=False, rnn_type='QGRU', layer_sizes=(64, 64), **kwargs):
        super(MultiLayerQRN, self).__init__()
        self.bidirectional = bidirectional
        self.rnn_type      = rnn_type
        layers = []
        prev_size = input_size
        for size in layer_sizes[:-1]:
            if self.bidirectional:
                layer = BidirectionalQRN(input_size=prev_size, hidden_size=size, rnn_type=self.rnn_type, **kwargs)
                prev_size = size * 2
            else:
                layer = QRN(input_size=prev_size, hidden_size=size, rnn_type=self.rnn_type, **kwargs)
                prev_size = size
            layers.append(layer)
            

        if 'dropout' in kwargs:
            del kwargs['dropout']
        if self.bidirectional:
            layer = BidirectionalQRN(input_size=prev_size, hidden_size=layer_sizes[-1], rnn_type=self.rnn_type,
                                     dropout=0.0, **kwargs)
        else:
            layer = QRN(input_size=prev_size, hidden_size=layer_sizes[-1], rnn_type=self.rnn_type,
                        dropout=0.0, **kwargs)

        layers.append(layer)
        self.layers = layers
        self.layer_sizes = layer_sizes
        self.input_size = input_size
        self.params = nn.ModuleList(layers)

    def reset_parameters(self):
        for l in self.layers:
            l.reset_parameters()

    def forward(self, x, hiddens):
        new_hiddens = []
        for l, h in zip(self.layers, hiddens):
            x, new_h = l(x, h)
            new_hiddens.append(new_h)
        return x, new_hiddens



class QuaternionRotation(Module):
    r"""Applies a quaternion rotation transformation to the incoming data.
    Args:
        in_features:  size of each quaternion input sample. The effective number
            of hidden units for each of the real and imaginary inputs.
            The total effective number of input hidden units is 4 x in_features.
        out_features: size of each quaternion output sample. The effective number
            of hidden units for each of the real and imaginary outputs.
            The total effective number of output hidden units is 4 x out_features.
        bias: If set to False, the layer will not learn an additive bias.
            Default: True
    Shape:
        - Input:  (N, 4 * in_features) 
        - Output: (N, 4 * out_features)
    Attributes:
        weight: the learnable weights of the module of shape
            (out_features x in_features)
        bias:   the learnable bias of the module of shape (2 * out_features)
    Examples::
        >>> m = QuaternionRotation(20, 30)
        >>> input = Variable(torch.randn(128, 80))
        >>> output = m(input)
        >>> print(output.size())
    """

    def __init__(self, in_features, out_features, bias=True,
                 init_criterion='glorot', weight_init='unitary',
                 seed=None):
        super(QuaternionLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r_weight = Parameter(torch.Tensor(in_features, out_features))
        self.i_weight = Parameter(torch.Tensor(in_features, out_features))
        self.j_weight = Parameter(torch.Tensor(in_features, out_features))
        self.k_weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(4 * out_features))
        else:
            self.register_parameter('bias', None)
        self.init_criterion = init_criterion
        self.weight_init = weight_init
        self.seed = seed if seed is not None else 1337
        self.rng = RandomState(self.seed)
        self.reset_parameters()

    def reset_parameters(self):
        winit = {'quaternion': quaternion_init,
                 'unitary': unitary_init}[self.weight_init]
        affect_init(self.r_weight, self.i_weight, self.j_weight, self.k_weight, winit,
                    self.rng, self.init_criterion)

    #### Double Real, is this a bug ?
    def forward(self, input):
        return quaternion_rotation(input, self.r_weight, self.i_weight, self.j_weight, self.k__weight, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', bias=' + str(self.bias is not None) \
            + ', init_criterion=' + str(self.init_criterion) \
            + ', weight_init=' + str(self.weight_init) \
            + ', seed=' + str(self.seed) + ')'



class QuaternionLinear(Module):
    r"""Applies a quaternion linear transformation to the incoming data.
    Args:
        in_features:  size of each quaternion input sample. The effective number
            of hidden units for each of the real and imaginary inputs.
            The total effective number of input hidden units is 4 x in_features.
        out_features: size of each quaternion output sample. The effective number
            of hidden units for each of the real and imaginary outputs.
            The total effective number of output hidden units is 4 x out_features.
        bias: If set to False, the layer will not learn an additive bias.
            Default: True
    Shape:
        - Input:  (N, 4 * in_features) 
        - Output: (N, 4 * out_features)
    Attributes:
        weight: the learnable weights of the module of shape
            (out_features x in_features)
        bias:   the learnable bias of the module of shape (2 * out_features)
    Examples::
        >>> m = QuaternionLinear(20, 30)
        >>> input = Variable(torch.randn(128, 80))
        >>> output = m(input)
        >>> print(output.size())
    """

    def __init__(self, in_features, out_features, bias=True,
                 init_criterion='glorot', weight_init='unitary',
                 seed=None):
        super(QuaternionLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r_weight = Parameter(torch.Tensor(in_features, out_features))
        self.i_weight = Parameter(torch.Tensor(in_features, out_features))
        self.j_weight = Parameter(torch.Tensor(in_features, out_features))
        self.k_weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(4 * out_features))
        else:
            self.register_parameter('bias', None)
        self.init_criterion = init_criterion
        self.weight_init = weight_init
        self.seed = seed if seed is not None else 1337
        self.rng = RandomState(self.seed)
        self.reset_parameters()

    def reset_parameters(self):
        winit = {'quaternion': quaternion_init,
                 'unitary': unitary_init}[self.weight_init]
        affect_init(self.r_weight, self.i_weight, self.j_weight, self.k_weight, winit,
                    self.rng, self.init_criterion)

    #### Double Real, is this a bug ?
    def forward(self, input):
        return quaternion_linear(input, self.r_weight, self.i_weight, self.j_weight, self.k__weight, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', bias=' + str(self.bias is not None) \
            + ', init_criterion=' + str(self.init_criterion) \
            + ', weight_init=' + str(self.weight_init) \
            + ', seed=' + str(self.seed) + ')'
