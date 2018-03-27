import numpy as np
from numpy.random import RandomState
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import Module
from complexops import *


def recurrent(inner, input, hx, reverse=False, **params):
    output = []
    steps = range(input.size(0) - 1, -1, -1) if reverse else range(input.size(0))
    hy = hx
    if inner in {ConvLSTMCell, CLSTMCell}:
        hy, cy = hx
    for i in steps:
        if inner in {ConvLSTMCell, CLSTMCell}:
            hy, cy = inner(input[i], (hy, cy), **params)
        else:
            hy = inner(input[i], hy, **params)
        output.append(hy)

    if reverse:
        output.reverse()
    output = torch.stack(output).squeeze(1)  # because output is a list containing
    return output, hy


# Heavily copied from https://github.com/pytorch/pytorch/blob/master/torch/nn/_functions/rnn.py#L47
def ConvGRUCell(input, hidden, w_ih, w_hh, b_ih, b_hh,
                dropout, train, rng, **opargs):

    if   input.dim() == 3:
        convop = F.conv1d
    elif input.dim() == 4:
        convop = F.conv2d
    elif input.dim() == 5:
        convop = F.conv3d
    else:
        raise Exception("The convolutional input is either 3, 4 or 5 dimensions."
                        " input.dim = " + str(input.dim()))

    do_dropout = train and dropout > 0.0

    h = hidden.squeeze(0)
    gi = convop(input,  w_ih, b_ih, **opargs)
    gh = convop(h,      w_hh, b_hh, **opargs)
    i_r, i_i, i_n = gi.chunk(3, 1)
    h_r, h_i, h_n = gh.chunk(3, 1)

    resetgate = F.sigmoid(i_r + h_r)
    inputgate = F.sigmoid(i_i + h_i)
    newgate   = F.tanh(i_n + resetgate * h_n)

    hy = newgate + inputgate * (h - newgate)

    return F.dropout2d(hy, p=dropout, training=do_dropout)


def ConvLSTMCell(input, hidden, w_ih, w_hh, b_ih, b_hh,
                 dropout, train, rng, **opargs):
    if   input.dim() == 3:
        convop = F.conv1d
    elif input.dim() == 4:
        convop = F.conv2d
    elif input.dim() == 5:
        convop = F.conv3d
    else:
        raise Exception("The convolutional input is either 3, 4 or 5 dimensions."
                        " input.dim = " + str(input.dim()))

    do_dropout = train and dropout > 0.0
    
    hx, cx = hidden
    h = hx.squeeze(0)
    c = cx.squeeze(0)

    gates = convop(input, w_ih, b_ih, **opargs) + convop(h, w_hh, b_hh, **opargs)

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate     = F.sigmoid(ingate)
    forgetgate = F.sigmoid(forgetgate)
    cellgate   = F.tanh(   cellgate)
    outgate    = F.sigmoid(outgate)

    cy = (forgetgate * c) + (ingate * cellgate)
    hy = outgate * F.tanh(cy)

    return F.dropout2d(hy, p=dropout, training=do_dropout), cy


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
                rt * torch.cat([ct_real_inp + ct_real_hid, ct_imag_inp + ct_imag_hid], dim=1))

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

    cy = (forgetgate * c) + (ingate * cellgate)
    hy = outgate * F.tanh(cy)

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


def OldCRUCell(input, hidden, w_ih_real, w_ih_imag,
               w_hh_real, w_hh_imag, wpick_cand_real,
               wpick_cand_imag, wpick_h_real, wpick_h_imag,
               b_ih, b_hh, b_pick_h, b_pick_cand,
               dropout, train, rng, dropout_type='complex',
               operation='linear', **opargs):

    if operation not in {'convolution', 'linear'}:
        raise Exception("the operations performed are either 'convolution' or 'linear'."
                        " Found operation = " + str(operation))

    complexop = complex_conv if operation == 'convolution' else complex_linear

    do_dropout = train and dropout > 0.0

    h = hidden.squeeze(0)
    h = get_normalized(h, input_type=operation)

    inp_amp_candidate = complexop(input,  w_ih_real, w_ih_imag, b_ih, **opargs)
    hid_amp_candidate = complexop(h,      w_hh_real, w_hh_imag, b_hh, **opargs)
    amp_and_candidate = inp_amp_candidate + hid_amp_candidate

    amp_real, candidate_real, amp_imag, candidate_imag = amp_and_candidate.chunk(4, 1)
    amp         = torch.cat([amp_real, amp_imag], dim=1)
    candidate   = torch.cat([candidate_real, candidate_imag], dim=1)
    if operation == 'linear':
        ampgate = get_modulus(amp, vector_form=True)
        candidate_t = F.tanh(candidate) * ampgate.repeat(1, 2)
    else:  # (which means if operation is convolutional)
        ampgate = get_modulus(amp, input_type='convolution')
        ampgate = ampgate.repeat(1, 2)
        if   candidate.dim() == 3:
            ampgate = ampgate.unsqueeze(-1).expand_as(candidate)
        elif candidate.dim() == 4:
            ampgate = ampgate.unsqueeze(-1).unsqueeze(-1).expand_as(candidate)
        elif candidate.dim() == 5:
            ampgate = ampgate.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(candidate)
        else:
            raise RuntimeError(
                "complex convolution accepts only input of dimension 3, 4 or 5."
                " candidate.dim = " + str(candidate.dim())
            )
        candidate_t = F.tanh(candidate) * ampgate

    candidate_t = get_normalized(candidate_t, input_type=operation)

    hidden_pre_update    = complexop(h,           wpick_h_real,    wpick_h_imag,       b_pick_h,    **opargs)
    candidate_pre_update = complexop(candidate_t, wpick_cand_real, wpick_cand_imag,    b_pick_cand, **opargs)

    update_hid_candidate = hidden_pre_update + candidate_pre_update
    (hidden_update_real, candidate_update_real,
     hidden_update_imag, candidate_update_imag) = update_hid_candidate.chunk(4, 1)
    hidden_update        = torch.cat([hidden_update_real, hidden_update_imag], dim=1)
    candidate_update     = torch.cat([candidate_update_real, candidate_update_imag], dim=1)

    if operation == 'linear':
        hidden_update_amp    = get_modulus(hidden_update,    vector_form=True)
        candidate_update_amp = get_modulus(candidate_update, vector_form=True)
        new_h = h * hidden_update_amp.repeat(1, 2) + candidate_t * candidate_update_amp.repeat(1, 2)
    else:  # (which means if operation is convolutional)
        hidden_update_amp = get_modulus(hidden_update, input_type='convolution')
        hidden_update_amp = hidden_update_amp.repeat(1, 2)
        if   h.dim() == 3:
            hidden_update_amp = hidden_update_amp.unsqueeze(-1).expand_as(h)
        elif h.dim() == 4:
            hidden_update_amp = hidden_update_amp.unsqueeze(-1).unsqueeze(-1).expand_as(h)
        elif h.dim() == 5:
            hidden_update_amp = hidden_update_amp.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(h)
        else:
            raise RuntimeError(
                "complex convolution accepts only input of dimension 3, 4 or 5."
                " h.dim = " + str(h.dim())
            )

        candidate_update_amp = get_modulus(candidate_update, input_type='convolution')
        candidate_update_amp = candidate_update_amp.repeat(1, 2)
        if   candidate_t.dim() == 3:
            candidate_update_amp = candidate_update_amp.unsqueeze(-1).expand_as(candidate_t)
        elif candidate_t.dim() == 4:
            candidate_update_amp = candidate_update_amp.unsqueeze(-1).unsqueeze(-1).expand_as(candidate_t)
        elif candidate_t.dim() == 5:
            candidate_update_amp = candidate_update_amp.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(candidate_t)
        else:
            raise RuntimeError(
                "complex convolution accepts only input of dimension 3, 4 or 5."
                " candidate_t.dim = " + str(candidate_t.dim())
            )
        
        new_h = h * hidden_update_amp + candidate_t * candidate_update_amp

    return apply_complex_dropout(new_h, dropout, rng, do_dropout, dropout_type, operation)
