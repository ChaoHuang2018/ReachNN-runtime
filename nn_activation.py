import numpy as np
import itertools
import math
import random
import time
import copy


class Activation(object):

    def __init__(
        self,
        activation,
        input,
        interval
    ):
        # neural networks
        self.activation = activation
        self.input = input
        self.interval = interval

        if self.activation == 'softplus':
            self.value = softplus(self.input)
            self.de = softplus_de(self.input)
            self.de2 = softplus_de2(self.input)
            self.output_range = softplus_range(self.interval)
            self.de_range = softplus_de_range(self.interval)
            self.de2_range = softplus_de2_range(self.interval)
        elif self.activation == 'sigmoid':
            self.value = sigmoid(self.input)
            self.de = sigmoid_de(self.input)
            self.de2 = sigmoid_de2(self.input)
            self.output_range = sigmoid_range(self.interval)
            self.de_range = sigmoid_de_range(self.interval)
            self.de2_range = sigmoid_de2_range(self.interval)
        elif self.activation == 'tanh':
            self.value = tanh(self.input)
            self.de = tanh_de(self.input)
            self.de2 = tanh_de2(self.input)
            self.output_range = tanh_range(self.interval)
            self.de_range = tanh_de_range(self.interval)
            self.de2_range = tanh_de2_range(self.interval)
        elif self.activation == 'Affine':
            self.value = affine(self.input)
            self.de = affine_de(self.input)
            self.de2 = affine_de2(self.input)
            self.output_range = affine_range(self.interval)
            self.de_range = affine_de_range(self.interval)
            self.de2_range = affine_de2_range(self.interval)

# define relu activation function and its left/right derivative
def softplus(x):
    return np.log(1+np.exp(x))


def softplus_de(x):
    return 1/(1+np.exp(-x))


def softplus_de2(x):
    return softplus_de(x)*(1-softplus_de(x))


def softplus_range(interval):
    return [softplus(interval[0]), softplus(interval[1])]


def softplus_de_range(interval):
    return [softplus_de(interval[0]), softplus_de(interval[1])]


def softplus_de2_range(interval):
    if interval[0] <= 0 and interval[1] >= 0:
        check_list = [interval[0], 0, interval[1]]
    low = 10000
    upp = -10000
    for i in range(len(check_list)):
        if softplus_de2(check_list[i]) <= softplus_de2(low):
            low = check_list[i]
        if softplus_de2(check_list[i]) >= softplus_de2(upp):
            upp = check_list[i]
    return [softplus_de2(low), softplus_de2(upp)]


# define tanh activation function and its left/right derivative
def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def tanh_de(x):
    return 1 - (tanh(x)) ** 2


def tanh_de2(x):
    return -2*tanh(x)*(1-tanh(x)**2)


def tanh_range(interval):
    return [tanh(interval[0]), tanh(interval[1])]


def tanh_de_range(interval):
    if interval[0] <= 0 and interval[1] >= 0:
        check_list = [interval[0], 0, interval[1]]
    low = 10000
    upp = -10000
    for i in range(len(check_list)):
        if tanh_de(check_list[i]) <= tanh_de(low):
            low = check_list[i]
        if tanh_de(check_list[i]) >= tanh_de(upp):
            upp = check_list[i]
    return [tanh_de(low), tanh_de(upp)]


def tanh_de2_range(interval):
    check_list = [interval[0], interval[1]]
    if interval[0] <= -np.sqrt(3)/3 and interval[1] >= -np.sqrt(3)/3:
        check_list.append(-np.sqrt(3)/3)
    if interval[0] <= np.sqrt(3)/3 and interval[1] >= np.sqrt(3)/3:
        check_list.append(np.sqrt(3)/3)
    low = 10000
    upp = -10000
    for i in range(len(check_list)):
        if tanh_de2(check_list[i]) <= tanh_de2(low):
            low = check_list[i]
        if tanh_de2(check_list[i]) >= tanh_de2(upp):
            upp = check_list[i]
    return [tanh_de2(low), tanh_de2(upp)]


# define sigmoid activation function and its left/right derivative
def sigmoid(x):
    return 1/(1+np.exp(-x))


def sigmoid_de(x):
    return sigmoid(x)*(1-sigmoid(x))


def sigmoid_de2(x):
    return 2*sigmoid(x)**3-3*sigmoid(x)**2+sigmoid(x)


def sigmoid_range(interval):
    return [sigmoid(interval[0]), sigmoid(interval(1))]


def sigmoid_de_range(interval):
    if interval[0] <= 0 and interval[1] >= 0:
        check_list = [interval[0], 0, interval[1]]
    low = 10000
    upp = -10000
    for i in range(len(check_list)):
        if sigmoid_de(check_list[i]) <= sigmoid_de(low):
            low = check_list[i]
        if sigmoid_de(check_list[i]) >= sigmoid_de(upp):
            upp = check_list[i]
    return [sigmoid_de(low), sigmoid_de(upp)]

def sigmoid_de2_range(interval):
    check_list = [interval[0], interval[1]]
    if interval[0] <= np.log(2-np.sqrt(3)) and interval[1] >= np.log(2-np.sqrt(3)):
        check_list.append(np.log(2-np.sqrt(3)))
    if interval[0] <= np.log(2+np.sqrt(3)) and interval[1] >= np.log(2+np.sqrt(3)):
        check_list.append(np.log(2+np.sqrt(3)))
    low = 10000
    upp = -10000
    for i in range(len(check_list)):
        if sigmoid_de2(check_list[i]) <= sigmoid_de2(low):
            low = check_list[i]
        if sigmoid_de2(check_list[i]) >= sigmoid_de2(upp):
            upp = check_list[i]
    return [sigmoid_de2(low), sigmoid_de2(upp)]

# define Indentity activation function and its left/right derivative
def affine(x):
    return x


def affine_de(x):
    return 1


def affine_de2(x):
    return 0


def affine_range(interval):
    return interval

def affine_de_range(interval):
    return [1,1]

def affine_de2_range(interval):
    return [0,0]