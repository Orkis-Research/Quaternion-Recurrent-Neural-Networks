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
        The code is equivalent of doing the following (Hamilton product):
        >>> input_r = get_r(input)
        >>> input_i = get_i(input)
        >>> input_j = get_j(input)
        >>> input_k = get_k(input)
        >>> r = input_r.mm(r_weight) - input_i.mm(i_weight) - input_j.mm(i_weight) - input_k.mm(i_weight)
        >>> i = input_i.mm(i_weight) + input_r.mm(r_weight) - input_k.mm(k_weight) + input_j.mm(j_weight)
        >>> j = input_j.mm(j_weight) + input_k.mm(k_weight) + input_r.mm(r_weight) - input_i.mm(i_weight)
        >>> k = input_k.mm(k_weight) - input_j.mm(j_weight) + input_i.mm(i_weight) + input_r.mm(r_weight)

        >>> if bias is not None:
        >>>    return torch.cat([r, i, j, k], dim=1) + bias
        >>> else:
        >>>    return torch.cat([r, i, j, k], dim=1)
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

#### Why unitary ?
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
    #Make these purely imaginary quaternions unitary
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
