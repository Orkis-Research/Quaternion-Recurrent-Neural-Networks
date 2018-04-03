import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from numpy.random import RandomState


def check_input(input):

    if input.dim() not in {2, 3}:
        raise RuntimeError(
            "quaternion linear accepts only input of dimension 2 or 3."
            " input.dim = " + str(input.dim())
        )

    nb_hidden = input.size()[-1]
    
    if nb_hidden % 4 != 0:
        raise RuntimeError(
            "Quaternion Tensors must be divisible by 4."
            " input.size()[1] = " + str(nb_hidden)
        )

######## Getters

def get_r(input):
    check_input(input)
    nb_hidden = input.size()[-1]
    if input.dim() == 2:
        return input.narrow(1, 0, nb_hidden // 4) # input[:, :nb_hidden / 2]
    elif input.dim() == 3:
        return input.narrow(2, 0, nb_hidden // 4) # input[:, :, :nb_hidden / 2]


def get_i(input):
    check_input(input)
    nb_hidden = input.size()[-1]
    if input.dim() == 2:
        return input.narrow(1, nb_hidden // 4, nb_hidden // 4) # input[:, nb_hidden / 2:]
    if input.dim() == 3:
        return input.narrow(2, nb_hidden // 4, nb_hidden // 4) # input[:, :, nb_hidden / 2:]

def get_j(input):
    check_input(input)
    nb_hidden = input.size()[-1]
    if input.dim() == 2:
        return input.narrow(1, nb_hidden // 2, nb_hidden // 4) # input[:, nb_hidden / 2:]
    if input.dim() == 3:
        return input.narrow(2, nb_hidden // 2, nb_hidden // 4) # input[:, :, nb_hidden / 2:]

def get_k(input):
    check_input(input)
    nb_hidden = input.size()[-1]
    if input.dim() == 2:
        return input.narrow(1, nb_hidden - nb_hidden // 4, nb_hidden // 4) # input[:, nb_hidden / 2:]
    if input.dim() == 3:
        return input.narrow(2, nb_hidden - nb_hidden // 4, nb_hidden // 4) # input[:, :, nb_hidden / 2:]


def get_modulus(input, vector_form=False):
    check_input(input)
    r = get_r(input)
    i = get_i(input)
    j = get_j(input)
    k = get_k(input)
    if vector_form:
        return torch.sqrt(r * r + i * i + j * j + k * k)
    else:
        return torch.sqrt((r * r + i * i + j * j + k * k).sum(dim=0))


def get_normalized(input, eps=0.0001):
    check_input(input)
    data_modulus = get_modulus(input)
    if input.dim() == 2:
        data_modulus_repeated = data_modulus.repeat(1, 4)
    elif input.dim() == 3:
        data_modulus_repeated = data_modulus.repeat(1, 1, 4)
    return input / (data_modulus_repeated.expand_as(input) + eps)

def quaternion_rotation(input, r_weight, i_weight, j_weight, k_weight, bias=None):
    """
    Applies a quaternion rotation (WIW^T) transformation to the incoming data:
    Shape:
        - Input:       (batch_size, nb_quaternion_elements_in * 4)
        - real_weight: (nb_quaternion_elements, nb_quaternion_elements_out)
        - imag_weight: (nb_quaternion_elements, nb_quaternion_elements_out)
        - Bias:        (nb_quaternion_elements_out * 4)
        - Output:      (batch_size, nb_quaternion_elements_out * 4)
    """

    #### ADD A CHECK TO SEE IF I IS PURELY IMAGINARY

    check_input(input)
    cat_kernels_4_r = torch.cat([r_weight, -i_weight, -j_weight, -k_weight], dim=0)
    cat_kernels_4_i = torch.cat([i_weight,  r_weight, -k_weight, j_weight], dim=0)
    cat_kernels_4_j = torch.cat([j_weight,  k_weight, r_weight, -i_weight], dim=0)
    cat_kernels_4_k = torch.cat([k_weight,  -j_weight, i_weight, r_weight], dim=0)
    cat_kernels_4_quaternion = torch.cat([cat_kernels_4_r, cat_kernels_4_i, cat_kernels_4_j, cat_kernels_4_k], dim=1)
    
    cat_kernels_4_rT = torch.cat([r_weight, i_weight, j_weight, k_weight], dim=0)
    cat_kernels_4_iT = torch.cat([-i_weight,  r_weight, k_weight, -j_weight], dim=0)
    cat_kernels_4_jT = torch.cat([-j_weight,  -k_weight, r_weight, i_weight], dim=0)
    cat_kernels_4_kT = torch.cat([-k_weight,  j_weight, -i_weight, r_weight], dim=0)
    cat_kernels_4_quaternion_T = torch.cat([cat_kernels_4_r, cat_kernels_4_i, cat_kernels_4_j, cat_kernels_4_k], dim=1)
    
    #### Woat ?
    if bias is not None:
        # fused op is marginally faster
        if input.dim() == 3:
            if input.size()[0] != 1:
                raise RuntimeError(
                    "Time dimension of the input different than 1."
                    " input.dim = " + str(input.dim())
                )
            input = input.squeeze(0)
        return torch.addmm(bias, cat_kernels_4_quaternion.mm(input), cat_kernels_4_quaternion_T)
    else:
        return cat_kernels_4_quaternion.mm(input).mm(cat_kernels_4_quaternion_T)



def quaternion_linear(input, r_weight, i_weight, j_weight, k_weight, bias=None):
    """
    Applies a quaternion linear transformation to the incoming data:
    Shape:
        - Input:       (batch_size, nb_quaternion_elements_in * 4)
        - real_weight: (nb_quaternion_elements, nb_quaternion_elements_out)
        - imag_weight: (nb_quaternion_elements, nb_quaternion_elements_out)
        - Bias:        (nb_quaternion_elements_out * 4)
        - Output:      (batch_size, nb_quaternion_elements_out * 4)
    code:
    """

    check_input(input)
    cat_kernels_4_r = torch.cat([r_weight, -i_weight, -j_weight, -k_weight], dim=0)
    cat_kernels_4_i = torch.cat([i_weight,  r_weight, -k_weight, j_weight], dim=0)
    cat_kernels_4_j = torch.cat([j_weight,  k_weight, r_weight, -i_weight], dim=0)
    cat_kernels_4_k = torch.cat([k_weight,  -j_weight, i_weight, r_weight], dim=0)
    cat_kernels_4_quaternion = torch.cat([cat_kernels_4_r, cat_kernels_4_i, cat_kernels_4_j, cat_kernels_4_k], dim=1)
    #print "kernel "
    #print cat_kernels_4_quaternion

   # print cat_kernels_4_quaternion.size()
    #print input.size()
    if bias is not None:
        # fused op is marginally faster
        if input.dim() == 3:
            if input.size()[0] != 1:
                raise RuntimeError(
                    "Time dimension of the input different than 1."
                    " input.dim = " + str(input.dim())
                )
            input = input.squeeze(0)
        return torch.addmm(bias, input, cat_kernels_4_quaternion)
    else:
        return input.mm(cat_kernels_4_quaternion)


def hamilton_product(q0, q1):
    """
    Applies a Hamilton product q0 * q1:
    Shape:
        - q0, q1 should be (batch_size, quaternion_number)
        (rr' - xx' - yy' - zz')  + 
        (rx' + xr' + yz' - zy')i +
        (ry' - xz' + yr' + zx')j +
        (rz' + xy' - yx' + zr')k + 
    """

    # NEED TO CHECK THE SHAPE OF THE INPUT
    #check_input(input)
    

    q1_r = get_r(q1)
    q1_i = get_i(q1)
    q1_j = get_j(q1)
    q1_k = get_k(q1)

    # rr', xx', yy', and zz' 
    r_base = torch.mul(q0, q1)
    # (rr' - xx' - yy' - zz')
    r   = get_r(r_base) - get_i(r_base) - get_j(r_base) - get_k(r_base)

    # rx', xr', yz', and zy'
    i_base = torch.mul(q0, torch.cat([q1_i, q1_r, q1_k, q1_j], dim=1))
    # (rx' + xr' + yz' - zy')
    i   = get_r(i_base) + get_i(i_base) + get_j(i_base) - get_k(i_base)

    # ry', xz', yr', and zx'
    j_base = torch.mul(q0, torch.cat([q1_j, q1_k, q1_r, q1_i], dim=1))
    # (rx' + xr' + yz' - zy')
    j   = get_r(j_base) - get_i(j_base) + get_j(j_base) + get_k(j_base)

    # rz', xy', yx', and zr'
    k_base = torch.mul(q0, torch.cat([q1_k, q1_j, q1_i, q1_r], dim=1))
    # (rx' + xr' + yz' - zy')
    k   = get_r(k_base) + get_i(k_base) - get_j(k_base) + get_k(k_base)

    return torch.cat([r, i, j, k], dim=1)


def unitary_init(in_features, out_features, rng, criterion='glorot'):
    
    if criterion == 'glorot':
        s = 1. / np.sqrt(2*(in_features + out_features))
    elif criterion == 'he':
        s = 1. / np.sqrt(2*in_features)
    else:
        raise ValueError('Invalid criterion: ' + criterion)
    kernel_shape = (in_features, out_features)
    number_of_weights = np.prod(kernel_shape) 
    v_r = np.random.uniform(0.0,1.0,number_of_weights)
    v_i = np.random.uniform(0.0,1.0,number_of_weights)
    v_j = np.random.uniform(0.0,1.0,number_of_weights)
    v_k = np.random.uniform(0.0,1.0,number_of_weights)
    #Make these unitary quaternion
    for i in range(0, number_of_weights):
    	norm = np.sqrt(v_r[i]**2 + v_i[i]**2 + v_j[i]**2 + v_k[i]**2)+0.0001
    	v_r[i]/= norm
        v_i[i]/= norm
        v_j[i]/= norm
        v_k[i]/= norm
    v_r = v_r.reshape(kernel_shape)
    v_i = v_i.reshape(kernel_shape)
    v_j = v_j.reshape(kernel_shape)
    v_k = v_k.reshape(kernel_shape)

    weight_r = v_r * s
    weight_i = v_i * s
    weight_j = v_j * s
    weight_k = v_k * s
    return (weight_r, weight_i, weight_j, weight_k)


def quaternion_init(in_features, out_features, rng, criterion='glorot'):
    

    if criterion == 'glorot':
        s = 1. / np.sqrt(2*(in_features + out_features))
    elif criterion == 'he':
        s = 1. / np.sqrt(2*in_features)
    else:
        raise ValueError('Invalid criterion: ' + criterion)
    rng = RandomState(seed)
    
    #Generating randoms and purely imaginary quaternions :
    kernel_shape = (in_features, out_features)
    number_of_weights = np.prod(kernel_shape) 
    v_i = np.random.uniform(0.0,1.0,number_of_weights)
    v_j = np.random.uniform(0.0,1.0,number_of_weights)
    v_k = np.random.uniform(0.0,1.0,number_of_weights)
    #Make these purely imaginary quaternions unitary
    for i in range(0, number_of_weights):
    	norm = np.sqrt(v_i[i]**2 + v_j[i]**2 + v_k[i]**2)+0.0001
    	v_i[i]/= norm
    	v_j[i]/= norm
    	v_k[i]/= norm
    v_i = v_i.reshape(kernel_shape)
    v_j = v_j.reshape(kernel_shape)
    v_k = v_k.reshape(kernel_shape)

    rng = RandomState(self.seed)
    modulus = rng.rayleigh(scale=s, size=kernel_shape)
    phase = rng.uniform(low=-np.pi, high=np.pi, size=kernel_shape)

    weight_r = modulus * np.cos(phase)
    weight_i = modulus * v_i*np.sin(phase)
    weight_j = modulus * v_j*np.sin(phase)
    weight_k = modulus * v_k*np.sin(phase)

    return (weight_r, weight_i, weight_j, weight_k)

def create_dropout_mask(dropout_p, size, rng, as_type, operation='linear'):
    if operation == 'linear':
        mask = rng.binomial(n=1, p=1-dropout_p, size=size)
        return Variable(torch.from_numpy(mask).type(as_type))
    else:
         raise Exception("create_dropout_mask accepts only 'linear'. Found operation = "
                        + str(operation))   


def apply_quaternion_mask(input, mask, dropout_type='quaternion', operation='linear'):
    if dropout_type == 'quaternion':
        input_r_masked = get_real(input, input_type=operation) * mask
        input_i_masked = get_imag(input, input_type=operation) * mask
        input_j_masked = get_imag(input, input_type=operation) * mask
        input_k_masked = get_imag(input, input_type=operation) * mask
        return torch.cat([input_r_masked, input_i_masked, input_j_masked, input_k_masked], dim=1)
    elif dropout_type == 'regular':
        return input * mask
    else:
        raise Exception("dropout_type accepts only 'complex' or 'regular'. Found dropout_type = "
                        + str(dropout_type))


def apply_quaternion_dropout(input, dropout_p, rng, do_dropout=True, dropout_type='quaternion', operation='linear'):
    size = input.data.size()
    s = []
    for i in range(input.dim()):
        s.append(size[i])
    if dropout_type == 'quaternion':
            s[1] = s[1] // 4
    elif dropout_type != 'regular':
        raise Exception("dropout_type accepts only 'quaternion' or 'regular'. Found dropout_type = "
                        + str(dropout_type))
    s = tuple(s)
    mask = create_dropout_mask(dropout_p, s, rng, input.data.type(), operation)
    return apply_complex_mask(input, mask, dropout_type, operation) / (1 - dropout_p) if do_dropout else input


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
