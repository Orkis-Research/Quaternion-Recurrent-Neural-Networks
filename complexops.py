import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from numpy.random import RandomState

def check_input(input):
    if input.dim() not in {2, 3}:
        raise Exception(
            "complex linear accepts only input of dimension 2 or 3."
            " input.dim = " + str(input.dim())
        )

    nb_hidden = input.size()[-1]

    if nb_hidden % 2 != 0:
        raise Exception(
            "complex Tensors have to have an even number of hidden dimensions."
            " input.size()[1] = " + str(nb_hidden)
        )

def check_conv_input(input):
    if input.dim() not in {3, 4, 5}:
        raise Exception(
            "complex convolution accepts only input of dimension 3, 4 or 5."
            " input.dim = " + str(input.dim())
        )

    nb_channels = input.size()[1]

    if nb_channels % 2 != 0:
        raise Exception(
            "complex Tensors have to have an even number of feature maps."
            " input.size()[1] = " + str(nb_channels)
        )


def get_real(input, input_type='linear'):
    if   input_type == 'linear':
        check_input(input)
    elif input_type == 'convolution':
        check_conv_input(input)
    else:
        raise Exception("the input_type can be either 'convolution' or 'linear'."
                        " Found input_type = " + str(input_type))

    if input_type == 'linear':
        nb_hidden = input.size()[-1]
        if input.dim() == 2:
            return input.narrow(1, 0, nb_hidden // 2) # input[:, :nb_hidden / 2]
        elif input.dim() == 3:
            return input.narrow(2, 0, nb_hidden // 2) # input[:, :, :nb_hidden / 2]
    else:
        nb_featmaps = input.size()[1]
        return input.narrow(1, 0, nb_featmaps // 2)


def get_imag(input, input_type='linear'):
    if   input_type == 'linear':
        check_input(input)
    elif input_type == 'convolution':
        check_conv_input(input)
    else:
        raise Exception("the input_type can be either 'convolution' or 'linear'."
                        " Found input_type = " + str(input_type))

    if input_type == 'linear':
        nb_hidden = input.size()[-1]
        if input.dim() == 2:
            return input.narrow(1, nb_hidden // 2, nb_hidden // 2) # input[:, :nb_hidden / 2]
        elif input.dim() == 3:
            return input.narrow(2, nb_hidden // 2, nb_hidden // 2) # input[:, :, :nb_hidden / 2]
    else:
        nb_featmaps = input.size()[1]
        return input.narrow(1, nb_featmaps // 2, nb_featmaps // 2)


def get_modulus(input, vector_form=False, input_type='linear'):
    if   input_type == 'linear':
        check_input(input)
        a = get_real(input)
        b = get_imag(input)
        if vector_form:
            return torch.sqrt(a * a + b * b)
        else:
            return torch.sqrt((a * a + b * b).sum(dim=-1))
    elif input_type == 'convolution':
        check_conv_input(input)
        a = get_real(input, input_type='convolution')
        b = get_imag(input, input_type='convolution')
        # we will return the modulus of each feature map
        return torch.sqrt((a * a + b * b).view(a.size(0), a.size(1), -1).sum(dim=-1))


def get_normalized(input, eps=0.001, threshold=1, input_type='linear'):
    if   input_type == 'linear':
        check_input(input)
    elif input_type == 'convolution':
        check_conv_input(input)
    else:
        raise Exception("the input_type can be either 'convolution' or 'linear'."
                        " Found input_type = " + str(input_type))

    if input_type == 'linear':
        data_modulus = get_modulus(input)
        m = data_modulus.unsqueeze(1).expand_as(input)
        mask = m < threshold
        return (input / (m + eps)) * (1 - mask.type_as(input)) + input * mask.type_as(input)
    else: # (which means if operation is convolutional)
        data_modulus = get_modulus(input, input_type='convolution')
        data_modulus = data_modulus.repeat(1, 2)
        if   input.dim() == 3:
            data_modulus = data_modulus.view(data_modulus.size(0), data_modulus.size(1), 1).expand_as(input)
        elif input.dim() == 4:
            data_modulus = data_modulus.view(data_modulus.size(0), data_modulus.size(1), 1, 1).expand_as(input)
        elif input.dim() == 5:
            data_modulus = data_modulus.view(data_modulus.size(0), data_modulus.size(1), 1, 1, 1).expand_as(input)
        
        mask = data_modulus < threshold
        return (input / (data_modulus + eps)) * (1 - mask.type_as(input)) + input * mask.type_as(input)


def complex_linear(input, real_weight, imag_weight, bias=None, **kwargs):
    """
    Applies a complex linear transformation to the incoming data:
    Shape:
        - Input:       (batch_size, nb_complex_elements_in * 2)
        - real_weight: (nb_complex_elements, nb_complex_elements_out)
        - imag_weight: (nb_complex_elements, nb_complex_elements_out)
        - Bias:        (nb_complex_elements_out * 2)
        - Output:      (batch_size, nb_complex_elements_out * 2)
    code:
        The code is equivalent of doing the following:
        >>> input_real = get_real(input)
        >>> input_imag = get_imag(input)
        >>> r = input_real.mm(real_weight) - input_imag.mm(imag_weight)
        >>> i = input_real.mm(imag_weight) + input_imag.mm(real_weight)

        >>> if bias is not None:
        >>>    return torch.cat([r, i], dim=1) + bias
        >>> else:
        >>>    return torch.cat([r, i], dim=1)
    """

    check_input(input)
    cat_kernels_4_real = torch.cat([real_weight, -imag_weight], dim=0)
    cat_kernels_4_imag = torch.cat([imag_weight,  real_weight], dim=0)
    cat_kernels_4_complex = torch.cat([cat_kernels_4_real, cat_kernels_4_imag], dim=1)
    if bias is not None:
        if input.dim() == 3:
            if input.size()[0] != 1:
                raise Exception(
                    "Time dimension of the input different than 1."
                    " input.dim = " + str(input.dim())
                )
            input = input.squeeze(0)
        return torch.addmm(bias, input, cat_kernels_4_complex)
    else:
        return input.mm(cat_kernels_4_complex)


def complex_conv(input, real_weight, imag_weight, bias=None, **convargs):
    """
    Applies a complex convolution to the incoming data:
    Shape:
        - Input:       (batch_size, nb_complex_channels_in * 2, *signal_length)
        - real_weight: (nb_complex_channels_out, nb_complex_channels_in, *kernel_size)
        - imag_weight: (nb_complex_channels_out, nb_complex_channels_in, *kernel_size)
        - Bias:        (nb_complex_channels_out * 2)
        - Output:      (batch_size, nb_complex_channels_out * 2, *signal_out_length)
        - convArgs =   {"strides":        strides,
                        "padding":        padding,
                        "dilation_rate":  dilation}
    """
    check_conv_input(input)
    cat_kernels_4_real = torch.cat([real_weight, -imag_weight], dim=1)
    cat_kernels_4_imag = torch.cat([imag_weight,  real_weight], dim=1)
    cat_kernels_4_complex = torch.cat([cat_kernels_4_real, cat_kernels_4_imag], dim=0)
    if   input.dim() == 3:
        convfunc = F.conv1d
    elif input.dim() == 4:
        convfunc = F.conv2d
    elif input.dim() == 5:
        convfunc = F.conv3d
    else:
        raise Exception("The convolutional input is either 3, 4 or 5 dimensions."
                        " input.dim = " + str(input.dim()))
    return convfunc(input, cat_kernels_4_complex, bias, **convargs)


def unitary_init(in_features, out_features, rng, criterion='glorot'):
    r = rng.uniform(size=(in_features, out_features))
    i = rng.uniform(size=(in_features, out_features))
    z = r + 1j * i
    u, _, v = np.linalg.svd(z)
    num_rows = in_features
    num_cols = out_features
    unitary_z = np.dot(u, np.dot(np.eye(int(num_rows), int(num_cols)), np.conjugate(v).T))
    indep_real = unitary_z.real
    indep_imag = unitary_z.imag
    if criterion == 'glorot':
        desired_var = 1. / (in_features + out_features)
    elif criterion == 'he':
        desired_var = 1. / (in_features)
    else:
        raise ValueError('Invalid criterion: ' + criterion)

    multip_real = np.sqrt(desired_var / np.var(indep_real))
    multip_imag = np.sqrt(desired_var / np.var(indep_imag))
    weight_real = multip_real * indep_real
    weight_imag = multip_imag * indep_imag

    return (weight_real, weight_imag)


def complex_init(in_features, out_features, rng, kernel_size=None, criterion='glorot'):
    if kernel_size is not None:
        receptive_field = np.prod(kernel_size)
        fan_out = out_features * receptive_field
        fan_in  = in_features  * receptive_field
    else:
        fan_out = out_features
        fan_in  = in_features
    if criterion == 'glorot':
        s = 1. / (fan_in + fan_out)
    elif criterion == 'he':
        s = 1. / fan_in
    else:
        raise ValueError('Invalid criterion: ' + criterion)
    size = (in_features, out_features) if kernel_size is None else (out_features, in_features) + tuple(kernel_size)
    modulus = rng.rayleigh(scale=s,                size=size)
    phase   = rng.uniform (low=-np.pi, high=np.pi, size=size)
    weight_real = modulus * np.cos(phase)
    weight_imag = modulus * np.sin(phase)

    return (weight_real, weight_imag)


def independent_complex_filters_init(in_channels, out_channels, kernel_size, rng, criterion='glorot'):
    if kernel_size is not None:
        num_rows = out_channels * in_channels
        num_cols = np.prod(kernel_size)
    else:
        num_rows = in_channels
        num_cols = out_channels
    flat_shape = (int(num_rows), int(num_cols))
    r = rng.uniform(size=flat_shape)
    i = rng.uniform(size=flat_shape)
    z = r + 1j * i
    u, _, v = np.linalg.svd(z)
    unitary_z = np.dot(u, np.dot(np.eye(int(num_rows), int(num_cols)), np.conjugate(v).T))
    real_unitary = unitary_z.real
    imag_unitary = unitary_z.imag
    if kernel_size is not None:
        indep_real = np.reshape(real_unitary, (num_rows,) + (kernel_size,))
        indep_imag = np.reshape(imag_unitary, (num_rows,) + (kernel_size,))
    else:
        indep_real = np.reshape(real_unitary, (num_rows, num_cols))
        indep_imag = np.reshape(imag_unitary, (num_rows, num_cols))

    receptive_field = num_cols
    if kernel_size is not None:
        fan_out = out_channels * receptive_field
        fan_in  = in_channels  * receptive_field
    else:
        fan_out = out_channels
        fan_in  = in_channels
    if   criterion == 'glorot':
        desired_var = 1. / (fan_in + fan_out)
    elif criterion == 'he':
        desired_var = 1. / (fan_in)
    else:
        raise ValueError('Invalid criterion: ' + self.criterion)
    multip_real  =  np.sqrt(desired_var / np.var(indep_real))
    multip_imag  =  np.sqrt(desired_var / np.var(indep_imag))
    scaled_real  =  multip_real * indep_real
    scaled_imag  =  multip_imag * indep_imag
    if kernel_size is not None:
        kernel_shape =  (int(out_channels), int(in_channels)) + (kernel_size,)
    else:
        kernel_shape =  (int(out_channels), int(in_channels))
    weight_real  =  np.reshape(scaled_real, kernel_shape)
    weight_imag  =  np.reshape(scaled_imag, kernel_shape)

    return (weight_real, weight_imag)

def affect_init(real_weight, imag_weight, init_func, rng, init_criterion):
    if real_weight.size() != imag_weight.size():
         raise ValueError('The real and imaginary weights '
                          'should have the same size . Found: '
                          + str(real_weight.size()) + ' and '
                          + str(imag_weight.size()))
    elif real_weight.dim() != 2:
        raise Exception('affect_init accepts only matrices. Found dimension = '
                        + str(real_weight.dim()))

    a, b = init_func(real_weight.size(0), real_weight.size(1), rng, init_criterion)
    a, b = torch.from_numpy(a), torch.from_numpy(b)
    real_weight.data = a.type_as(real_weight.data)
    imag_weight.data = b.type_as(imag_weight.data)


def affect_conv_init(real_weight, imag_weight, kernel_size, init_func, rng, init_criterion):
    if real_weight.size() != imag_weight.size():
         raise ValueError('The real and imaginary weights '
                          'should have the same size . Found: '
                          + str(real_weight.size()) + ' and '
                          + str(imag_weight.size()))
    elif 2 >= real_weight.dim():
        raise Exception('affect_conv_init accepts only tensors that have more than 2 dimensions. Found dimension = '
                        + str(real_weight.dim()))

    a, b = init_func(real_weight.size(1), real_weight.size(0), kernel_size, rng, init_criterion)
    a, b = torch.from_numpy(a), torch.from_numpy(b)
    real_weight.data = a.type_as(real_weight.data)
    imag_weight.data = b.type_as(imag_weight.data)


def create_dropout_mask(dropout_p, size, rng, as_type, operation='linear'):
    if operation == 'linear':
        mask = rng.binomial(n=1, p=1-dropout_p, size=size)
        return Variable(torch.from_numpy(mask).type(as_type))
    elif operation == 'convolution':
        mask_size_expanded    = [1] * len(size)
        mask_size_expanded[0] = size[0]
        mask_size_expanded[1] = size[1]
        mask = rng.binomial(n=1, p=1-dropout_p, size=tuple(mask_size_expanded)) * np.ones((size))
        return Variable(torch.from_numpy(mask).type(as_type))
    else:
         raise Exception("create_dropout_mask accepts only 'linear' or 'convolution' as operation. Found operation = "
                        + str(operation))   


def apply_complex_mask(input, mask, dropout_type='complex', operation='linear'):
    if dropout_type == 'complex':
        input_real_masked = get_real(input, input_type=operation) * mask
        input_imag_masked = get_imag(input, input_type=operation) * mask
        return torch.cat([input_real_masked, input_imag_masked], dim=1)
    elif dropout_type == 'regular':
        return input * mask
    else:
        raise Exception("dropout_type accepts only 'complex' or 'regular'. Found dropout_type = "
                        + str(dropout_type))


def apply_complex_dropout(input, dropout_p, rng, do_dropout=True, dropout_type='complex', operation='linear'):
    size = input.data.size()
    s = []
    for i in range(input.dim()):
        s.append(size[i])
    if dropout_type == 'complex':
            s[1] = s[1] // 2
    elif dropout_type != 'regular':
        raise Exception("dropout_type accepts only 'complex' or 'regular'. Found dropout_type = "
                        + str(dropout_type))
    s = tuple(s)
    mask = create_dropout_mask(dropout_p, s, rng, input.data.type(), operation)
    return apply_complex_mask(input, mask, dropout_type, operation) / (1 - dropout_p) if do_dropout else input
