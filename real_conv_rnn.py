from numpy.random import RandomState
import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn import Module
from torch import nn
from realops import *
from rnn import recurrent, ConvGRUCell, ConvLSTMCell


class ConvolutionalRNN(Module):

    def __init__(self, in_channels, out_channels, kernel_size, rec_kernel_size,
                 stride, dilation, padding, rnn_type='GRU', bias=True, dropout=0,
                 reverse=False, init_criterion='glorot', weight_init='orthogonal',
                 seed=None, **kwargs):

        super(ConvolutionalRNN, self).__init__(**kwargs)
        self.in_channels     =    in_channels
        self.out_channels    =    out_channels
        self.kernel_size     =    kernel_size
        self.rec_kernel_size =    rec_kernel_size
        self.stride          =    stride
        self.padding         =    padding
        self.dilation        =    dilation
        self.bias            =    bias
        self.dropout         =    dropout
        self.reverse         =    reverse
        self.init_criterion  =    init_criterion
        self.weight_init     =    weight_init
        self.seed            =    seed if seed is not None else 1337
        self.rng             =    RandomState(self.seed)
        self.winit           =    {'orthogonal': independent_filters_init}[self.weight_init]
        self.rnn_type        =    rnn_type

        if self.rnn_type == 'GRU':
            gate_channels    =    3 * self.out_channels
        if self.rnn_type == 'LSTM':
            gate_channels    =    4 * self.out_channels

        # define parameters:
        self.w_ih_shape      =    (gate_channels, self.in_channels)   + tuple((self.kernel_size,))
        self.w_hh_shape      =    (gate_channels, self.out_channels)  + tuple((self.rec_kernel_size,))
        
        self.w_ih            =    Parameter(torch.Tensor(*self.w_ih_shape))
        self.w_hh            =    Parameter(torch.Tensor(*self.w_hh_shape))
        
        if self.bias:
            self.b_ih        =    Parameter(torch.Tensor(gate_channels))
            self.b_hh        =    Parameter(torch.Tensor(gate_channels))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

        self.reset_parameters()

    def reset_parameters(self):
        fargs = [self.winit, self.rng, self.init_criterion]
        affect_conv_init(self.w_ih,     self.kernel_size, *fargs)
        affect_conv_init(self.w_hh, self.rec_kernel_size, *fargs)

        if self.bias:
            self.b_ih.data.zero_()
            self.b_hh.data.zero_()

    def forward(self, input, hx):
        params = {
            'w_ih'        : self.w_ih,
            'w_hh'        : self.w_hh,
            'b_ih'        : self.b_ih,
            'b_hh'        : self.b_hh,
            'dropout'     : self.dropout,
            'train'       : self.training,
            'rng'         : self.rng
        }

        params.update({
            'stride'      : self.stride,
            'padding'     : self.padding,
            'dilation'    : self.dilation}
        )
        cellname = {'GRU': ConvGRUCell, 'LSTM': ConvLSTMCell}[self.rnn_type]
        return recurrent(cellname, input, hx, reverse=self.reverse, **params)


class BidirectionalConvolutionalRNN(Module):

    def __init__(self, in_channels, out_channels, kernel_size, rec_kernel_size,
                 stride, dilation, padding, rnn_type='GRU', bias=True, dropout=0,
                 init_criterion='glorot', weight_init='orthogonal', seed=None, **kwargs):

        super(BidirectionalConvolutionalRNN, self).__init__()
        self.rnn_type    =  rnn_type
        arguments = {
            'in_channels'     :   in_channels,    'out_channels'    : out_channels,
            'kernel_size'     :   kernel_size,    'rec_kernel_size' : rec_kernel_size,
            'stride'          :   stride,         'dilation'        : dilation,
            'padding'         :   padding,        'rnn_type'        : self.rnn_type,
            'bias'            :   bias,           'dropout'         : dropout,
            'init_criterion'  :   init_criterion, 'weight_init'     : weight_init,
            'seed'            :   seed
        }
        self.forwardcrn   =  ConvolutionalRNN(reverse=False, **dict(**arguments, **kwargs))
        self.backwardcrn  =  ConvolutionalRNN(reverse=True,  **dict(**arguments, **kwargs))

    def reset_parameters(self):
        self.forwardcrn.reset_parameters()
        self.backwardcrn.reset_parameters()

    def forward(self, input, hx):
        if   self.rnn_type in {'LSTM'}:
            o1, lasth1 = self.forwardcrn (input, (hx[0][0], hx[0][1]))
            o2, lasth2 = self.backwardcrn(input, (hx[1][0], hx[1][1]))
        elif self.rnn_type in {'GRU'}:
            o1, lasth1 = self.forwardcrn (input, hx[0].unsqueeze(0))
            o2, lasth2 = self.backwardcrn(input, hx[1].unsqueeze(0))
        out   = torch.cat([o1, o2], dim=2)
        hlast = torch.cat([o1[-1].unsqueeze(0), o2[-1].unsqueeze(0)], dim=0)
        return out, hlast


class MultiLayerConvolutionalRNN(Module):
    """heavily inspired from:
       https://github.com/pytorch/benchmark/blob/master/benchmarks/lstm_variants/container.py
    """
    def __init__(self, in_channels, bidirectional=False, rnn_type='GRU', out_channels_list=(64, 64), **kwargs):
        super(MultiLayerConvolutionalRNN, self).__init__()
        self.bidirectional = bidirectional
        self.rnn_type      = rnn_type
        layers = []
        prev_channels = in_channels
        for out_channels in out_channels_list[:-1]:
            if self.bidirectional:
                layer = BidirectionalConvolutionalRNN(
                    in_channels=prev_channels,
                    out_channels=out_channels,
                    rnn_type=self.rnn_type,
                    **kwargs)
                prev_channels = out_channels * 2
            else:
                layer = ConvolutionalRNN(
                    in_channels=prev_channels,
                    out_channels=out_channels,
                    rnn_type=self.rnn_type,
                    **kwargs)
                prev_channels = out_channels

            layers.append(layer)

        if 'dropout' in kwargs:
            del kwargs['dropout']
        if self.bidirectional:
            layer = BidirectionalConvolutionalRNN(
                in_channels=prev_channels,
                out_channels=out_channels_list[-1],
                rnn_type=self.rnn_type,
                dropout=0,
                **kwargs)
            # HERE DROPOUT SHOULD BE EQUAL TO 0 AS IT IS THE LAST LAYER
        else:
            layer = ConvolutionalRNN(
                in_channels=prev_channels,
                out_channels=out_channels_list[-1],
                rnn_type=self.rnn_type,
                dropout=0,
                **kwargs)
        layers.append(layer)
        self.layers = layers
        self.out_channels_list = out_channels_list
        self.in_channels = in_channels
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
