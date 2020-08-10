import numpy as np
import tensorflow as tf
from itertools import product
import math
import random
import time
import copy
from nn_activation import Activation


class NeuronInfo(object):
    def __init__(
        self,
        nn_input_dim,
        activation_type,
        input_value=None,
        input_range=None,
        first_order_der_value=[],
        first_order_der_range=[],
        second_order_der_range=[]
    ):
        self.nn_input_dim = nn_input_dim
        self.input_value = input_value
        self.input_range = input_range
        self.first_order_der_value = first_order_der_value
        self.first_order_der_range = first_order_der_range
        self.second_order_der_range = second_order_der_range
        self.activation_type = activation_type

    def set_input_value(self, last_layer_info, weight, bias):
        if self.input_value is None:
            input_value = bias
            for j in range(len(last_layer_info)):
                input_value = input_value + weight[j] * last_layer_info[j].activation_info.value
            self.input_value = input_value

    def set_input_range(self, last_layer_info, weight, bias):
        if self.input_range is None:
            input_range_low = bias
            input_range_upp = bias
            for j in range(len(last_layer_info)):
                l_omega_low = last_layer_info[j].activation_info.output_range[0]
                l_omega_upp = last_layer_info[j].activation_info.output_range[1]
                if weight[j] < 0:
                    u_omega_low = l_omega_upp
                    u_omega_upp = l_omega_low
                else:
                    u_omega_low = l_omega_low
                    u_omega_upp = l_omega_upp
                input_range_low = input_range_low + weight[j] * u_omega_low
                input_range_upp = input_range_upp + weight[j] * u_omega_upp
            if input_range_low > input_range_upp:
                print('Interval fault!')
            self.input_range = [input_range_low, input_range_upp]


    def set_first_order_der_value(self, last_layer_info, weight):
        if not self.first_order_der_value:
            for i in range(self.nn_input_dim):
                first_order_der_value = 0
                for j in range(len(last_layer_info)):
                    first_order_der_value = first_order_der_value + weight[j] * last_layer_info[j].activation_info.de * \
                                            last_layer_info[j].first_order_der_value[i]
                self.first_order_der_value.append(first_order_der_value)

    def set_first_order_der_range(self, last_layer_info, weight):
        if not self.first_order_der_range:
            for i in range(self.nn_input_dim):
                first_order_der_low = 0
                first_order_der_upp = 0
                for j in range(len(last_layer_info)):
                    l_omega_low = min([m*n for m,n in product(last_layer_info[j].activation_info.de_range,last_layer_info[j].first_order_der_range[i])])
                    l_omega_upp = max([m*n for m,n in product(last_layer_info[j].activation_info.de_range,last_layer_info[j].first_order_der_range[i])])
                    if weight[j] < 0:
                        u_omega_low = l_omega_upp
                        u_omega_upp = l_omega_low
                    else:
                        u_omega_low = l_omega_low
                        u_omega_upp = l_omega_upp
                    first_order_der_low = first_order_der_low + weight[j] * u_omega_low
                    first_order_der_upp = first_order_der_upp + weight[j] * u_omega_upp
                if isinstance(first_order_der_low, (np.ndarray, list)):
                    first_order_der_low_num = first_order_der_low[0]
                else:
                    first_order_der_low_num = first_order_der_low
                if isinstance(first_order_der_upp, (np.ndarray, list)):
                    first_order_der_upp_num = first_order_der_upp[0]
                else:
                    first_order_der_upp_num = first_order_der_upp
                if first_order_der_low_num > first_order_der_upp_num:
                    print('first der fault!')
                self.first_order_der_range.append([first_order_der_low_num, first_order_der_upp_num])
            print('self.first_order_der_range: ' + str(self.first_order_der_range))


    def set_second_order_der_range(self, last_layer_info, weight):
        if not self.second_order_der_range:
            for i in range(self.nn_input_dim):
                second_order_der_low = 0
                second_order_der_upp = 0
                for j in range(len(last_layer_info)):
                    # print(j)
                    # if not last_layer_info[j].second_order_der_range:
                    #     print(last_layer_info[j].input_value)
                    #     print(last_layer_info[j].input_range)
                    #     print(last_layer_info[j].first_order_der_value)
                    #     print(last_layer_info[j].first_order_der_range)
                    #     print(last_layer_info[j].second_order_der_range)
                    l_omega_low = min([m*n+p for m,n,p in product(last_layer_info[j].activation_info.de2_range,last_layer_info[j].first_order_der_range[i],last_layer_info[j].second_order_der_range[i])])
                    l_omega_upp = max([m*n+p for m,n,p in product(last_layer_info[j].activation_info.de2_range,last_layer_info[j].first_order_der_range[i],last_layer_info[j].second_order_der_range[i])])
                    if weight[j] < 0:
                        u_omega_low = l_omega_upp
                        u_omega_upp = l_omega_low
                    else:
                        u_omega_low = l_omega_low
                        u_omega_upp = l_omega_upp
                    second_order_der_low = second_order_der_low + weight[j] * u_omega_low
                    second_order_der_upp = second_order_der_upp + weight[j] * u_omega_upp
                if isinstance(second_order_der_low, (np.ndarray, list)):
                    second_order_der_low_num = second_order_der_low[0]
                else:
                    second_order_der_low_num = second_order_der_low
                if isinstance(second_order_der_upp, (np.ndarray, list)):
                    second_order_der_upp_num = second_order_der_upp[0]
                else:
                    second_order_der_upp_num = second_order_der_upp
                if second_order_der_low_num > second_order_der_upp_num:
                    print('Second der fault!')
                self.second_order_der_range.append([second_order_der_low_num, second_order_der_upp_num])
            print('self.second_order_der_range: ' + str(self.second_order_der_range))

    def set_activation_info(self):
        self.activation_info = Activation(self.activation_type,self.input_value,self.input_range)

