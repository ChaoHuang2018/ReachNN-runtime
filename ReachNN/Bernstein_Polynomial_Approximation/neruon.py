import numpy as np
import tensorflow as tf
from itertools import product
import math
import random
import time
import copy
from nn_activation import Activation
from interval import interval, inf, imath

class NeuronInfo(object):
    def __init__(
        self,
        nn_input_dim,
        activation_type,
        input_value=None,
        input_range=None,
        first_order_der_value=[],
        first_order_der_low=[],
        first_order_der_upp=[],
        second_order_der_value=[],
        second_order_der_low=[],
        second_order_der_upp=[]
    ):
        self.nn_input_dim = nn_input_dim
        self.input_value = input_value
        self.input_range = input_range
        if first_order_der_value == []:
            self.first_order_der_value = np.zeros(self.nn_input_dim)
        else:
            self.first_order_der_value = first_order_der_value
        if first_order_der_low == []:
            self.first_order_der_low = np.zeros(self.nn_input_dim)
        else:
            self.first_order_der_low = first_order_der_low
        if first_order_der_upp == []:
            self.first_order_der_upp = np.zeros(self.nn_input_dim)
        else:
            self.first_order_der_upp = first_order_der_upp
        if second_order_der_value == []:
            self.second_order_der_value = np.zeros((self.nn_input_dim, self.nn_input_dim))
        else:
            self.second_order_der_value = second_order_der_value
        if second_order_der_low == []:
            self.second_order_der_low = np.zeros((self.nn_input_dim, self.nn_input_dim))
        else:
            self.second_order_der_low = second_order_der_low
        if second_order_der_upp == []:
            self.second_order_der_upp = np.zeros((self.nn_input_dim, self.nn_input_dim))
        else:
            self.second_order_der_upp = second_order_der_upp
        self.activation_type = activation_type

    def set_input_value(self, last_layer_info, weight, bias):
        if self.input_value is None:
            input_value = float(bias[0])
            for j in range(len(last_layer_info)):
                input_value = input_value + float(weight[j]) * last_layer_info[j].activation_info.value
            self.input_value = input_value

    def set_input_range(self, last_layer_info, weight, bias):
        if self.input_range is None:
            input_range = interval[0, 0] + float(bias[0])
            for j in range(len(last_layer_info)):
                input_range = input_range + float(weight[j]) * last_layer_info[j].activation_info.output_range
            self.input_range = input_range
            # input_range_low = bias
            # input_range_upp = bias
            # for j in range(len(last_layer_info)):
            #     l_omega_low = last_layer_info[j].activation_info.output_range[0].inf
            #     l_omega_upp = last_layer_info[j].activation_info.output_range[0].sup
            #     if weight[j] < 0:
            #         u_omega_low = l_omega_upp
            #         u_omega_upp = l_omega_low
            #     else:
            #         u_omega_low = l_omega_low
            #         u_omega_upp = l_omega_upp
            #     input_range_low = input_range_low + weight[j] * u_omega_low
            #     input_range_upp = input_range_upp + weight[j] * u_omega_upp
            # if input_range_low > input_range_upp:
            #     print('Interval fault!')
            # self.input_range = interval[input_range_low, input_range_upp]

    def set_first_order_der_value(self, last_layer_info, weight):
        for i in range(self.nn_input_dim):
            first_order_der_value_i = 0
            for j in range(len(last_layer_info)):
                first_order_der_value_i = first_order_der_value_i + float(weight[j]) * last_layer_info[j].activation_info.de * \
                                        last_layer_info[j].first_order_der_value[i]
            self.first_order_der_value[i] = first_order_der_value_i

    def set_first_order_der_range(self, last_layer_info, weight):
        for i in range(self.nn_input_dim):
            first_order_der_range = interval[0, 0]
            for j in range(len(last_layer_info)):
                j_first_order_der_range = interval[last_layer_info[j].first_order_der_low[i],
                                                   last_layer_info[j].first_order_der_upp[i]]
                first_order_der_range = first_order_der_range + float(weight[j]) * last_layer_info[
                    j].activation_info.de_range * j_first_order_der_range
            self.first_order_der_low[i] = first_order_der_range[0].inf
            self.first_order_der_upp[i] = first_order_der_range[0].sup
        # print('self.first_order_der_range: ' + str([self.first_order_der_low, self.first_order_der_upp]))

    def set_second_order_der_value(self, last_layer_info, weight):
        for i in range(self.nn_input_dim):
            for s in range(i, self.nn_input_dim):
                second_order_der_value = 0
                for j in range(len(last_layer_info)):
                    second_order_der_value = second_order_der_value + float(weight[j]) * last_layer_info[
                        j].activation_info.de2 * \
                                             last_layer_info[j].first_order_der_value[i] * \
                                             last_layer_info[j].first_order_der_value[s] + last_layer_info[
                                                 j].activation_info.de * last_layer_info[j].second_order_der_value[
                                                 i, s]
                self.second_order_der_value[i,s] = self.second_order_der_value[s,i] = second_order_der_value

    def set_second_order_der_range(self, last_layer_info, weight):
        for i in range(self.nn_input_dim):
            for s in range(i, self.nn_input_dim):
                second_order_der_range = interval[0, 0]
                for j in range(len(last_layer_info)):
                    j_i_first_order_der_range = interval[last_layer_info[j].first_order_der_low[i],
                                                         last_layer_info[j].first_order_der_upp[i]]
                    j_s_first_order_der_range = interval[last_layer_info[j].first_order_der_low[s],
                                                         last_layer_info[j].first_order_der_upp[s]]
                    j_i_s_second_order_der_range = interval[last_layer_info[j].second_order_der_low[i,s],
                                                         last_layer_info[j].second_order_der_upp[i,s]]

                    second_order_der_range = second_order_der_range + float(weight[j]) * (last_layer_info[
                                                                                       j].activation_info.de2_range * j_i_first_order_der_range * j_s_first_order_der_range +
                                                                                   last_layer_info[
                                                                                       j].activation_info.de_range *
                                                                                   j_i_s_second_order_der_range)
                    self.second_order_der_low[i,s] = self.second_order_der_low[s,i] = second_order_der_range[0].inf
                    self.second_order_der_upp[i, s] = self.second_order_der_upp[s, i] = second_order_der_range[0].sup
        #print('self.second_order_der_range: ' + str(self.second_order_der_range))

    def set_activation_info(self):
        self.activation_info = Activation(self.activation_type,self.input_value,self.input_range)

