import numpy as np
import itertools
import math
import random
import time
import copy
from interval import interval, inf, imath
import sys

class Activation(object):

    def __init__(
        self,
        activation,
        input,
        interval_bound
    ):
        # neural networks
        self.activation = activation
        self.input = input
        self.interval_bound = interval_bound

        if self.activation == 'softplus':
            self.value = softplus(self.input)
            self.de = softplus_de(self.input)
            self.de2 = softplus_de2(self.input)
            self.output_range = softplus_range(self.interval_bound)
            self.de_range = softplus_de_range(self.interval_bound)
            self.de2_range = softplus_de2_range(self.interval_bound)
        elif self.activation == 'sigmoid':
            self.value = sigmoid(self.input)
            self.de = sigmoid_de(self.input)
            self.de2 = sigmoid_de2(self.input)
            self.output_range = sigmoid_range(self.interval_bound)
            self.de_range = sigmoid_de_range(self.interval_bound)
            self.de2_range = sigmoid_de2_range(self.interval_bound)
        elif self.activation == 'tanh':
            self.value = tanh(self.input)
            self.de = tanh_de(self.input)
            self.de2 = tanh_de2(self.input)
            self.output_range = tanh_range(self.interval_bound)
            self.de_range = tanh_de_range(self.interval_bound)
            self.de2_range = tanh_de2_range(self.interval_bound)
        elif self.activation == 'Affine':
            self.value = affine(self.input)
            self.de = affine_de(self.input)
            self.de2 = affine_de2(self.input)
            self.output_range = affine_range(self.interval_bound)
            self.de_range = affine_de_range(self.interval_bound)
            self.de2_range = affine_de2_range(self.interval_bound)

        if self.value not in self.output_range:
            sys.exit("This is the error: output/ouput range not match")
        if self.de not in self.de_range:
            sys.exit("This is the error: de/de range not match")
        if self.de2 not in self.de2_range:
            sys.exit("This is the error: de2/de2 range not match")

# define relu activation function and its left/right derivative
def softplus(x):
    return np.log(1+np.exp(x))


def softplus_de(x):
    return 1/(1+np.exp(-x))


def softplus_de2(x):
    return softplus_de(x)*(1-softplus_de(x))


def softplus_range(interval_bound):
    return interval[softplus(interval_bound[0].inf), softplus(interval_bound[0].sup)]


def softplus_de_range(interval_bound):
    return interval[softplus_de(interval_bound[0].inf), softplus_de(interval_bound[0].sup)]


def softplus_de2_range(interval_bound):
    if interval_bound[0].inf <= 0 and interval_bound[0].sup >= 0:
        check_list = [interval_bound[0].inf, 0, interval_bound[0].sup]
    else:
        check_list = [interval_bound[0].inf, interval_bound[0].sup]
    low = 10000
    upp = -10000
    for i in range(len(check_list)):
        if softplus_de2(check_list[i]) <= softplus_de2(low):
            low = check_list[i]
        if softplus_de2(check_list[i]) >= softplus_de2(upp):
            upp = check_list[i]
    return interval[softplus_de2(low), softplus_de2(upp)]


# define tanh activation function and its left/right derivative
def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def tanh_de(x):
    return 1 - (tanh(x)) ** 2


def tanh_de2(x):
    return -2*tanh(x)*(1-tanh(x)**2)


def tanh_range(interval_bound):
    return interval[tanh(interval_bound[0].inf), tanh(interval_bound[0].sup)]


def tanh_de_range(interval_bound):
    if interval_bound[0].inf <= 0 and interval_bound[0].sup >= 0:
        check_list = [interval_bound[0].inf, 0, interval_bound[0].sup]
    else:
        check_list = [interval_bound[0].inf, interval_bound[0].sup]
    low = 10000
    upp = -10000
    for i in range(len(check_list)):
        if tanh_de(check_list[i]) <= tanh_de(low):
            low = check_list[i]
        if tanh_de(check_list[i]) >= tanh_de(upp):
            upp = check_list[i]
    return interval[tanh_de(low), tanh_de(upp)]


def tanh_de2_range(interval_bound):
    check_list = [interval_bound[0], interval_bound[1]]
    if interval_bound[0].inf <= -np.sqrt(3)/3 and interval_bound[0].sup >= -np.sqrt(3)/3:
        check_list.append(-np.sqrt(3)/3)
    if interval_bound[0].inf <= np.sqrt(3)/3 and interval_bound[0].sup >= np.sqrt(3)/3:
        check_list.append(np.sqrt(3)/3)
    low = 10000
    upp = -10000
    for i in range(len(check_list)):
        if tanh_de2(check_list[i]) <= tanh_de2(low):
            low = check_list[i]
        if tanh_de2(check_list[i]) >= tanh_de2(upp):
            upp = check_list[i]
    return interval[tanh_de2(low), tanh_de2(upp)]


# define sigmoid activation function and its left/right derivative
def sigmoid(x):
    if x < 0:
        return 1. - 1. / (1. + np.exp(x))
    else:
        return 1. / (1. + np.exp(-x))


def sigmoid_de(x):
    return sigmoid(x)*(1-sigmoid(x))


def sigmoid_de2(x):
    return 2*sigmoid(x)**3-3*sigmoid(x)**2+sigmoid(x)


def sigmoid_range(interval_bound):
    return interval[sigmoid(interval_bound[0].inf), sigmoid(interval_bound[0].sup)]


def sigmoid_de_range(interval_bound):
    if interval_bound[0].inf <= 0 and interval_bound[0].sup >= 0:
        check_list = [interval_bound[0].inf, 0, interval_bound[0].sup]
    else:
        check_list = [interval_bound[0].inf, interval_bound[0].sup]
    low = 10000
    upp = -10000
    for i in range(len(check_list)):
        if sigmoid_de(check_list[i]) <= sigmoid_de(low):
            low = check_list[i]
        if sigmoid_de(check_list[i]) >= sigmoid_de(upp):
            upp = check_list[i]
    return interval[sigmoid_de(low), sigmoid_de(upp)]

def sigmoid_de2_range(interval_bound):
    check_list = [interval_bound[0].inf, interval_bound[0].sup]
    if interval_bound[0].inf <= np.log(2-np.sqrt(3)) and interval_bound[0].sup >= np.log(2-np.sqrt(3)):
        check_list.append(np.log(2-np.sqrt(3)))
    if interval_bound[0].inf <= np.log(2+np.sqrt(3)) and interval_bound[0].sup >= np.log(2+np.sqrt(3)):
        check_list.append(np.log(2+np.sqrt(3)))
    low = 10000
    upp = -10000
    for i in range(len(check_list)):
        if sigmoid_de2(check_list[i]) <= sigmoid_de2(low):
            low = check_list[i]
        if sigmoid_de2(check_list[i]) >= sigmoid_de2(upp):
            upp = check_list[i]
    return interval[sigmoid_de2(low), sigmoid_de2(upp)]

# define Indentity activation function and its left/right derivative
def affine(x):
    return x


def affine_de(x):
    return 1


def affine_de2(x):
    return 0


def affine_range(interval_bound):
    return interval_bound

def affine_de_range(interval_bound):
    return interval[1,1]

def affine_de2_range(interval_bound):
    return interval[0,0]