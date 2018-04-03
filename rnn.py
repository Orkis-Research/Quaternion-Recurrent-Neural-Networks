import numpy as np
from numpy.random import RandomState
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import Module
from complexops import *
from quaternionops import *

def recurrent(inner, input, hx, reverse=False, **params):
    output = []
    steps = range(input.size(0) - 1, -1, -1) if reverse else range(input.size(0))
    hy = hx
    if inner in {CLSTMCell, QLSTMCell}:
        hy, cy = hx
    for i in steps:
        if inner in {QLSTMCell, CLSTMCell}:
            hy, cy = inner(input[i], (hy, cy), **params)
        else:
            hy = inner(input[i], hy, **params)
        output.append(hy)

    if reverse:
        output.reverse()
    output = torch.stack(output).squeeze(1)  # because output is a list containing
    return output, hy


#
# COMPLEX
#


def CGRUCell(input, hidden, w_ih_real, w_ih_imag, w_hh_real, w_hh_imag,
             b_ih, b_hh, dropout, train, rng, dropout_type='complex',
             operation='linear', **opargs):

    if operation not in {'convolution', 'linear'}:
        raise Exception("the operations performed are either 'convolution' or 'linear'."
                        " Found operation = " + str(operation))

    complexop = complex_conv if operation == 'convolution' else complex_linear

    do_dropout = train and dropout > 0.0

    h = hidden.squeeze(0)
    zt_rt_ct_inp = complexop(input,  w_ih_real, w_ih_imag, b_ih, **opargs)
    zt_rt_ct_hid = complexop(h,      w_hh_real, w_hh_imag, b_hh, **opargs)
    zt_real_inp, rt_real_inp, ct_real_inp, zt_imag_inp, rt_imag_inp, ct_imag_inp = zt_rt_ct_inp.chunk(6, 1)
    zt_real_hid, rt_real_hid, ct_real_hid, zt_imag_hid, rt_imag_hid, ct_imag_hid = zt_rt_ct_hid.chunk(6, 1)
    zt = F.sigmoid(torch.cat([zt_real_inp + zt_real_hid, zt_imag_inp + zt_imag_hid], dim=1))
    rt = F.sigmoid(torch.cat([rt_real_inp + rt_real_hid, rt_imag_inp + rt_imag_hid], dim=1))
    ct = F.tanh(torch.cat([ct_real_inp, ct_imag_inp], dim=1) + \
               complex_product(rt, torch.cat([ct_real_inp + ct_real_hid, ct_imag_inp + ct_imag_hid], dim=1)))

    return apply_complex_dropout(ct + zt * (h - ct), dropout, rng, do_dropout, dropout_type, operation)


def CLSTMCell(input, hidden, w_ih_real, w_ih_imag, w_hh_real, w_hh_imag,
              b_ih, b_hh, dropout, train, rng, dropout_type='complex',
              operation='linear', **opargs):

    if operation not in {'convolution', 'linear'}:
        raise Exception("the operations performed are either 'convolution' or 'linear'."
                        " Found operation = " + str(operation))

    complexop = complex_conv if operation == 'convolution' else complex_linear

    do_dropout = train and dropout > 0.0

    hx, cx = hidden
    h = hx.squeeze(0)
    c = cx.squeeze(0)

    gates = complexop(input, w_ih_real, w_ih_imag, b_ih, **opargs) + complexop(h, w_hh_real, w_hh_imag, b_hh, **opargs)
    (ingate_real, forgetgate_real, cellgate_real, outgate_real,
     ingate_imag, forgetgate_imag, cellgate_imag, outgate_imag) = gates.chunk(8, 1)

    ingate     = F.sigmoid(torch.cat([    ingate_real,     ingate_imag], dim=1))
    forgetgate = F.sigmoid(torch.cat([forgetgate_real, forgetgate_imag], dim=1))
    cellgate   = F.tanh(   torch.cat([  cellgate_real,   cellgate_imag], dim=1))
    outgate    = F.sigmoid(torch.cat([   outgate_real,    outgate_imag], dim=1))

    
    fc = complex_product(forgetgate, c)
    ic = complex_product(ingate, cellgate)       

    cy = fc + ic
    hy = complex_product(outgate, F.tanh(cy))
    #cy = forgetgate * c + ingate * cellgate
    #hy = outgate * F.tanh(cy)
    return apply_complex_dropout(hy, dropout, rng, do_dropout, dropout_type, operation), cy


def CRUCell(input, hidden, w_ih_real, w_ih_imag,
            w_hh_real, w_hh_imag, b_ih, b_hh,
            w_ih_mod, w_hh_mod, b_ih_mod, b_hh_mod,
            dropout, train, rng, dropout_type='complex',
            operation='linear',
            modulation_activation='softmax', **opargs):

    if operation not in {'convolution', 'linear'}:
        raise Exception("the operations performed are either 'convolution' or 'linear'."
                        " Found operation = " + str(operation))

    complexop = complex_conv if operation == 'convolution' else complex_linear
    if operation == 'convolution':
        if   input.dim() == 3:
            modop = F.conv1d
        elif input.dim() == 4:
            modop = F.conv2d
        elif input.dim() == 5:
            modop = F.conv3d
        else:
            raise Exception("The convolutional input is either 3, 4 or 5 dimensions."
                        " input.dim = " + str(input.dim()))
    else:
        modop = F.linear

    do_dropout = train and dropout > 0.0

    h = hidden.squeeze(0)

    complex_gates = complexop(
        input, w_ih_real, w_ih_imag, b_ih, **opargs
    )             + complexop(
        h,w_hh_real, w_hh_imag, b_hh, **opargs
    )
    zt_real, ct_real, zt_imag, ct_imag = complex_gates.chunk(4, 1)
    zt = F.sigmoid(torch.cat([zt_real, zt_imag], dim=1))
    ct = torch.cat(          [ct_real, ct_imag], dim=1)

    if operation == 'linear':
        mod_gate        = modop(input, w_ih_mod, b_ih_mod) + modop(h, w_hh_mod, b_hh_mod)
    else:
        mod_gate        = modop(input, w_ih_mod, b_ih_mod, **opargs) + modop(h, w_hh_mod, b_hh_mod, **opargs)
    
    mod_act             = {'softmax'  : F.log_softmax,
                           'sigmoid'  : F.sigmoid,
                           'relu'     : F.relu,
                           'identity' : 'identity'}[modulation_activation]
    # Softmax corresponds to featurewise attention.
    if   modulation_activation == 'softmax':
        mod_gate = mod_act(mod_gate, dim=1).exp()
        # we use exponential(log_softmax(x)) because log_softmax is more stable.
        # It allows to avoid getting sums of zeros in the denominator of the softmax
        # see https://discuss.pytorch.org/t/how-to-avoid-nan-in-softmax/1676
        # see http://pytorch.org/docs/0.3.1/nn.html#torch.nn.functional.log_softmax
    elif modulation_activation != 'identity':
        mod_gate = mod_act(mod_gate)
    repeated_size       = mod_gate.dim() * [1]
    repeated_size[1]    = 2
    mod_gate            = mod_gate.repeat(*repeated_size)
    ct                  = F.tanh(mod_gate * ct)
    hy = ct + zt * (h - ct)

    return apply_complex_dropout(hy, dropout, rng, do_dropout, dropout_type, operation)

#
# QUATERNION
#

def QRNNBaseCell(input, hidden,  w_ih_r, w_ih_i, w_ih_j, w_ih_k,
                             w_hh_r, w_hh_i, w_hh_j, w_hh_k,
                             b_ih, b_hh, dropout, train, rng, 
                             dropout_type='quaternion',
                             operation='linear', **opargs):

    if operation not in {'rotation', 'linear'}:
        raise Exception("the operations performed are either 'rotation' or 'linear'."
                        " Found operation = " + str(operation))

    quatop = quaternion_rotation if operation == 'rotation' else quaternion_linear

    do_dropout = train and dropout > 0.0

    h = hidden.squeeze(0)
    st_rt_ct_inp = quatop(input,  w_ih_r, w_ih_i, w_ih_j, w_ih_k, b_ih, **opargs)
    st_rt_ct_hid = quatop(h,      w_hh_r, w_hh_i, w_hh_j, w_hh_k, b_hh, **opargs)
    
    (st_r_inp, st_i_inp, st_j_inp, st_k_inp) = st_rt_ct_inp.chunk(4, 1)
    (st_r_hid, st_i_hid, st_j_hid, st_k_hid) = st_rt_ct_hid.chunk(4, 1)
    
    st = F.tanh(torch.cat([st_r_inp + st_r_hid, st_i_inp + st_i_hid, st_j_inp + st_j_hid, st_k_inp + st_k_hid], dim=1))

    return apply_quaternion_dropout(st, dropout, rng, do_dropout, dropout_type, operation)

def QGRUCell(input, hidden,  w_ih_r, w_ih_i, w_ih_j, w_ih_k,
                             w_hh_r, w_hh_i, w_hh_j, w_hh_k,
                             b_ih, b_hh, dropout, train, rng, 
                             dropout_type='quaternion',
                             operation='linear', **opargs):

    if operation not in {'rotation', 'linear'}:
        raise Exception("the operations performed are either 'rotation' or 'linear'."
                        " Found operation = " + str(operation))

    quatop = quaternion_rotation if operation == 'rotation' else quaternion_linear

    do_dropout = train and dropout > 0.0

    h = hidden.squeeze(0)
    zt_rt_ct_inp = quatop(input,  w_ih_r, w_ih_i, w_ih_j, w_ih_k, b_ih, **opargs)
    zt_rt_ct_hid = quatop(h,      w_hh_r, w_hh_i, w_hh_j, w_hh_k, b_hh, **opargs)
    (zt_r_inp, rt_r_inp, ct_r_inp, 
    zt_i_inp, rt_i_inp, ct_i_inp, 
    zt_j_inp, rt_j_inp, ct_j_inp, 
    zt_k_inp, rt_k_inp, ct_k_inp) = zt_rt_ct_inp.chunk(12, 1)
    (zt_r_hid, rt_r_hid, ct_r_hid, 
    zt_i_hid, rt_i_hid, ct_i_hid, 
    zt_j_hid, rt_j_hid, ct_j_hid, 
    zt_k_hid, rt_k_hid, ct_k_hid) = zt_rt_ct_hid.chunk(12, 1)
    
    zt = F.sigmoid(torch.cat([zt_r_inp + zt_r_hid, zt_i_inp + zt_i_hid, zt_j_inp + zt_j_hid, zt_k_inp + zt_k_hid], dim=1))
    rt = F.sigmoid(torch.cat([rt_r_inp + rt_r_hid, rt_i_inp + rt_i_hid, rt_j_inp + rt_j_hid, rt_k_inp + rt_k_hid], dim=1))
    ct = F.tanh(torch.cat([ct_r_inp, ct_i_inp, ct_j_inp, ct_k_inp], dim=1) + \
                hamilton_product(rt, torch.cat([ct_r_inp + ct_r_hid, ct_i_inp + ct_i_hid, ct_j_inp + ct_j_hid, ct_k_inp + ct_k_hid], dim=1)))

    return apply_quaternion_dropout(ct + zt * (h - ct), dropout, rng, do_dropout, dropout_type, operation)

def QLSTMCell(input, hidden, w_ih_r, w_ih_i, w_ih_j, w_ih_k,
                             w_hh_r, w_hh_i, w_hh_j, w_hh_k,
                             b_ih, b_hh, dropout, train, rng, 
                             dropout_type='quaternion',
                             operation='linear', **opargs):

    if operation not in {'rotation', 'linear'}:
        raise Exception("the operations performed are either 'rotation' or 'linear'."
                        " Found operation = " + str(operation))

    quatop = quaternion_rotation if operation == 'rotation' else quaternion_linear

    do_dropout = train and dropout > 0.0

    hx, cx = hidden
    h = hx.squeeze(0)
    c = cx.squeeze(0)

    gates = quatop(input, w_ih_r, w_ih_i, w_ih_j, w_ih_k, b_ih, **opargs) + quatop(h, w_hh_r, w_hh_i, w_hh_j, w_hh_k, b_hh, **opargs)
    
    (ingate_r, forgetgate_r, cellgate_r, outgate_r,
     ingate_i, forgetgate_i, cellgate_i, outgate_i,
     ingate_j, forgetgate_j, cellgate_j, outgate_j,
     ingate_k, forgetgate_k, cellgate_k, outgate_k) = gates.chunk(16, 1)
      
    ingate     = F.sigmoid(torch.cat([    ingate_r,     ingate_i,     ingate_j,     ingate_k], dim=1))
    forgetgate = F.sigmoid(torch.cat([forgetgate_r, forgetgate_i, forgetgate_j, forgetgate_k], dim=1))
    cellgate   = F.tanh(   torch.cat([  cellgate_r,   cellgate_i,   cellgate_j,   cellgate_k], dim=1))
    outgate    = F.sigmoid(torch.cat([   outgate_r,    outgate_i,    outgate_j,    outgate_k], dim=1))

    #fc = hamilton_product(forgetgate, c)
    #ic = hamilton_product(ingate, cellgate)
    #cy = fc + ic
    cy = (forgetgate * c) + (ingate * cellgate)
    #hy = hamilton_product(outgate, F.tanh(cy))
    #cy = (forgetgate * c) + (ingate * cellgate)
    hy = outgate * F.tanh(cy)
    return apply_quaternion_dropout(hy, dropout, rng, do_dropout, dropout_type, operation), cy

