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
from quaternionops import quaternion_linear, quaternion_rotation, unitary_init, quaternion_init, get_r, get_i, get_j, get_k, get_normalized, get_modulus
import math
import sys

cpt_test = 0

def affect_init(r_weight, i_weight, j_weight, k_weight, init_func, rng, init_criterion):
    if r_weight.size() != i_weight.size() or r_weight.size() != j_weight.size() or \
    r_weight.size() != k_weight.size() :
         raise ValueError('The real and imaginary weights '
                 'should have the same size . Found: r:'
                 + str(r_weight.size()) +' i:'
                 + str(i_weight.size()) +' j:'
                 + str(j_weight.size()) +' k:'
                 + str(k_weight.size()))

    elif r_weight.dim() != 2:
        raise Exception('affect_init accepts only matrices. Found dimension = '
                        + str(r_weight.dim()))
    r, i, j, k = init_func(r_weight.size(0), r_weight.size(1), rng, init_criterion)
    r, i, j, k = torch.from_numpy(r), torch.from_numpy(i), torch.from_numpy(j), torch.from_numpy(k)
    r_weight.data = r.type_as(r_weight.data)
    i_weight.data = i.type_as(i_weight.data)
    j_weight.data = j.type_as(j_weight.data)
    k_weight.data = k.type_as(k_weight.data)


def create_dropout_mask(dropout_p, size, rng, as_type):
    mask = rng.binomial(n=1, p=1-dropout_p, size=size)
    return Variable(torch.from_numpy(mask).type(as_type))


def apply_quaternion_mask(input, mask):
    input_r_masked = get_r(input) * mask
    input_i_masked = get_i(input) * mask
    input_j_masked = get_j(input) * mask
    input_k_masked = get_k(input) * mask
    return torch.cat([input_r_masked, input_i_masked, input_j_masked, input_k_masked],
                     dim=-1)

def apply_quaternion_dropout(input, dropout_p, rng, do_dropout=True):
    size = input.data.size()
    s = []
    for i in range(input.dim()):
        s.append(size[i])
    s[-1] = s[-1] // 4
    s = tuple(s)
    mask = create_dropout_mask(
        dropout_p, s, rng, input.data.type()
    )
    return apply_quaternion_mask(input, mask) / (1 - dropout_p) if do_dropout else input

def QRUCell(input, hidden, w_ih_r, w_ih_i, w_ih_j, w_ih_k,
                           w_hh_r, w_hh_i, w_hh_j, w_hh_k, 
                           wpick_cand_r, wpick_cand_i, wpick_cand_j, wpick_cand_k, 
                           wpick_h_r, wpick_h_i, wpick_h_j, wpick_h_k,
                           b_ih, b_hh, b_pick_h, b_pick_cand,
                           dropout, train, rng):
    
    do_dropout = train and dropout > 0.0

    h = hidden.squeeze(0)
    h = get_normalized(h)
    
    inp_amp_candidate = quaternion_linear(input,  w_ih_r, w_ih_i, w_ih_j, w_ih_k, b_ih)
    hid_amp_candidate = quaternion_linear(h, w_hh_r, w_hh_i, w_hh_j, w_hh_k, b_hh)
    
    amp_and_candidate = inp_amp_candidate + hid_amp_candidate
    (amp_r, candidate_r, 
    amp_i, candidate_i, 
    amp_j, candidate_j, 
    amp_k, candidate_k) = amp_and_candidate.chunk(8, 1)
    amp       = torch.cat([amp_r, amp_i, amp_j, amp_k], dim=-1)
    candidate = torch.cat([candidate_r, candidate_i, candidate_j, candidate_k], dim=-1)
    ampgate = get_modulus(amp, vector_form=True)
    candidate_t = candidate * ampgate.repeat(1, 4)
    candidate_t = get_normalized(candidate_t)
    hidden_pre_update    = quaternion_linear(h,           wpick_h_r,    wpick_h_i,    wpick_h_j,    wpick_h_k,     b_pick_h)
    candidate_pre_update = quaternion_linear(candidate_t, wpick_cand_r, wpick_cand_i, wpick_cand_j, wpick_cand_k,  b_pick_cand)
    #print "hidden_pre_update"
    #print hidden_pre_update
    #print "h"
    #print h
    update_hid_candidate = hidden_pre_update + candidate_pre_update
    (hidden_update_r, candidate_update_r, 
     hidden_update_i, candidate_update_i,     
     hidden_update_j, candidate_update_j, 
     hidden_update_k, candidate_update_k)  = update_hid_candidate.chunk(8, 1)
    hidden_update        = torch.cat([hidden_update_r, hidden_update_i,
                                      hidden_update_i, hidden_update_j], dim=-1)
    candidate_update     = torch.cat([candidate_update_r, candidate_update_i,
                                      candidate_update_j, candidate_update_k], dim=-1)

    hidden_update_amp    = get_modulus(hidden_update,    vector_form=True)
    candidate_update_amp = get_modulus(candidate_update, vector_form=True)
    a = h * hidden_update_amp.repeat(1, 4) + candidate_t * candidate_update_amp.repeat(1, 4)
    #print "Input"
    #print input
    #print "inp amp"
    #print inp_amp_candidate
    #print "hidden"
    #print h
    #print "hid_amp"
    #print hid_amp_candidate

    return apply_quaternion_dropout(a, dropout, rng, do_dropout)


def recurrent(inner, input, hx, reverse=False, **params):
    output = []
    steps = range(input.size(0) - 1, -1, -1) if reverse else range(input.size(0))
    hy = hx
    for i in steps:
        hy = inner(input[i], hy, **params)
        output.append(hy)

    if reverse:
        output.reverse()
    output = torch.stack(output).squeeze(1)  # because output is a list containing
    return output, hy

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


class QRU(Module):
    r"""Applies a 1 layer quaternary recurrent unit (QRU) RNN to an input sequence.

    For each element in the input sequence, each layer computes the following
    function:

    Args:
        input_size: The number of expected quaternion features in the input x
        hidden_size: The number of quaternion features in the hidden state h
        bias: If ``False``, then the layer does not use bias weights b_ih and b_hh.
            Default: ``True``
        dropout: If non-zero, performs quaternion dropout on the outputs of each
            RNN layer except the last layer

    Inputs: input, h_0
        - **input** (seq_len, batch, 4 * input_size): quaternion tensor containing the features
          of the input sequence.
        - **h_0** (num_layers * num_directions, batch, 4 * hidden_size): quaternion tensor
          containing the initial hidden state for each element in the batch.

    Outputs: output, h_n
        - **output** (seq_len, batch, 4 * hidden_size * num_directions): tensor
          containing the output features h_t from the last layer of the RNN,
          for each t. If a :class:`torch.nn.utils.rnn.PackedSequence` has been
          given as the input, the output will also be a packed sequence.
        - **h_n** (num_layers * num_directions, batch, hidden_size): tensor
          containing the hidden state for t=seq_len

    """
    def __init__(self, input_size, hidden_size,
                 bias=True, dropout=0, reverse=False,
                 init_criterion='glorot',
                 weight_init='unitary',
                 seed=None, **kwargs):
        super(QRU, self).__init__(**kwargs)
        self.input_size      =    input_size
        self.hidden_size     =    hidden_size
        self.bias            =    bias
        self.dropout         =    dropout
        self.reverse         =    reverse
        self.init_criterion  =    init_criterion
        self.weight_init     =    weight_init
        self.seed            =    seed if seed is not None else 1337
        self.rng             =    RandomState(self.seed)
        self.winit           =    {'quaternion': quaternion_init,
                                  'unitary': unitary_init}[self.weight_init]
        
        # 2 because Two gates
        gate_size            =    2 * self.hidden_size

        ######## Define Cell Parameters :
        # Input to Hidden
        # Hidden to Hidden
        # ??
        # ??

        self.w_ih_r       =    Parameter(torch.Tensor(self.input_size // 4,  gate_size))
        self.w_ih_i       =    Parameter(torch.Tensor(self.input_size // 4,  gate_size))
        self.w_ih_j       =    Parameter(torch.Tensor(self.input_size // 4,  gate_size))
        self.w_ih_k       =    Parameter(torch.Tensor(self.input_size // 4,  gate_size))
        self.w_hh_r       =    Parameter(torch.Tensor(self.hidden_size, gate_size))
        self.w_hh_i       =    Parameter(torch.Tensor(self.hidden_size, gate_size))
        self.w_hh_j       =    Parameter(torch.Tensor(self.hidden_size, gate_size))
        self.w_hh_k       =    Parameter(torch.Tensor(self.hidden_size, gate_size))
        self.wpick_cand_r =    Parameter(torch.Tensor(self.hidden_size, gate_size))
        self.wpick_cand_i =    Parameter(torch.Tensor(self.hidden_size, gate_size))
        self.wpick_cand_j =    Parameter(torch.Tensor(self.hidden_size, gate_size))
        self.wpick_cand_k =    Parameter(torch.Tensor(self.hidden_size, gate_size))
        self.wpick_h_r    =    Parameter(torch.Tensor(self.hidden_size, gate_size))
        self.wpick_h_i    =    Parameter(torch.Tensor(self.hidden_size, gate_size))
        self.wpick_h_j    =    Parameter(torch.Tensor(self.hidden_size, gate_size))
        self.wpick_h_k    =    Parameter(torch.Tensor(self.hidden_size, gate_size))
        if self.bias:
            self.b_ih        =    Parameter(torch.Tensor(gate_size * 4))
            self.b_hh        =    Parameter(torch.Tensor(gate_size * 4))
            self.b_pick_h    =    Parameter(torch.Tensor(gate_size * 4))
            self.b_pick_cand =    Parameter(torch.Tensor(gate_size * 4))
        else:
            self.register_parameter('bias_ih',        None)
            self.register_parameter('bias_hh',        None)
            self.register_parameter('bias_pick_h',    None)
            self.register_parameter('bias_pick_cand', None)

        self.reset_parameters()

    def reset_parameters(self):
        fargs = [self.winit, self.rng, self.init_criterion]
        affect_init(self.w_ih_r,       self.w_ih_i,       self.w_ih_j,  self.w_ih_k, *fargs)
        affect_init(self.w_hh_r,       self.w_hh_i,       self.w_hh_j,  self.w_hh_k, *fargs)
        affect_init(self.wpick_cand_r, self.wpick_cand_i, self.wpick_cand_j, self.wpick_cand_k, *fargs)
        affect_init(self.wpick_h_r,    self.wpick_h_i,\
                    self.wpick_h_j,    self.wpick_h_k,    *fargs)
        if self.bias:
            self.b_ih.data.zero_()
            self.b_hh.data.zero_()
            self.b_pick_h.data.zero_()
            self.b_pick_cand.data.zero_()

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
            'wpick_cand_r': self.wpick_cand_r, 
            'wpick_cand_i': self.wpick_cand_i,
            'wpick_cand_j': self.wpick_cand_j,
            'wpick_cand_k': self.wpick_cand_k,
            'wpick_h_r': self.wpick_h_r,
            'wpick_h_i': self.wpick_h_i,
            'wpick_h_j': self.wpick_h_j,
            'wpick_h_k': self.wpick_h_k,
            'b_ih': self.b_ih,
            'b_hh': self.b_hh,
            'b_pick_h': self.b_pick_h,
            'b_pick_cand': self.b_pick_cand,
            'dropout': self.dropout,
            'train': self.training,
            'rng': self.rng

        }
        return recurrent(QRUCell, input, hx, reverse=self.reverse, **params)


class BidirectionalQRU(Module):

    def __init__(self, input_size, hidden_size,
                 bias=True, dropout=0,
                 init_criterion='glorot',
                 weight_init='unitary',
                 seed=None, **kwargs):

        super(BidirectionalQRU, self).__init__()
        self.forwardqru   = QRU(input_size, hidden_size,
                                bias=bias, dropout=dropout, reverse=False,
                                init_criterion=init_criterion,
                                weight_init=weight_init,
                                seed=seed, **kwargs)
        self.backwardqru  = QRU(input_size, hidden_size,
                                bias=bias, dropout=dropout, reverse=True,
                                init_criterion=init_criterion,
                                weight_init=weight_init,
                                seed=seed, **kwargs)

    def reset_parameters(self):
        self.forwardqru.reset_parameters()
        self.backwardqru.reset_parameters()

    def forward(self, input, hx):
        o1, lasth1 = self.forwardqru (input, hx[0].unsqueeze(0))
        o2, lasth2 = self.backwardqru(input, hx[1].unsqueeze(0))
        out   = torch.cat([o1, o2], dim=-1)
        hlast = torch.cat([o1[-1].unsqueeze(0), o2[-1].unsqueeze(0)], dim=0)
        return out, hlast


class MultiLayerQRU(Module):
    """heavily inspired from:
       https://github.com/pytorch/benchmark/blob/master/benchmarks/lstm_variants/container.py
    """

    #### why 64 ?
    def __init__(self, input_size, qru_type, layer_sizes=(64, 64), **kwargs):
        super(MultiLayerQRU, self).__init__()
        self.qru_type = qru_type
        layers = []
        prev_size = input_size
        for size in layer_sizes[:-1]:
            if self.qru_type == 'biQRU':
                layer = BidirectionalQRU(input_size=prev_size, hidden_size=size, **kwargs)
            elif self.qru_type == 'QRU':
                layer = QRU(input_size=prev_size, hidden_size=size, **kwargs)
            else:
                raise Exception('MultiLayerCRU accepts only CRU and biCRU as layer types'
                                'Found Layer type = ' + str(cru_type))
            layers.append(layer)
            prev_size = size * 4
            # prev_size is twice size because in the cru layer w_ih_real and _imag take
            # as input something that has size input_size // 2.

        if 'dropout' in kwargs:
            del kwargs['dropout']
        if self.qru_type == 'biQRU':
            layer = BidirectionalQRU(input_size=prev_size, hidden_size=layer_sizes[-1],
                                     dropout=0.0, **kwargs)
        elif self.qru_type == 'QRU':
            layer = QRU(input_size=prev_size, hidden_size=layer_sizes[-1],
                        dropout=0.0, **kwargs)
        else:
            raise Exception('MultiLayerQRU accepts only QRU and biQRU as layer types for now'
                            'Found Layer type = ' + str(qru_type))

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

