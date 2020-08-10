import numpy as np
import tensorflow as tf
import itertools
import math
import random
import time
import copy
from neruon import NeuronInfo
from numpy import linalg as LA

class NNRange(object):

    def __init__(
        self,
        NN
    ):
        # neural networks
        self.NN = NN
        print(self.NN.num_of_inputs)
        print(self.NN.activations)
        print('self.NN.network_structure' + str(self.NN.network_structure))

    def get_local_lips(self, network_input_box, output_index):
        # center
        center = [0.5 * (interval[0] + interval[1]) for interval in network_input_box]
        half_len = [0.5 * (interval[1] - interval[0]) for interval in network_input_box]

        weight_all_layer = self.NN.weights
        print('weight_all_layer: ' + str(len(weight_all_layer)))
        bias_all_layer = self.NN.bias
        print('bias_all_layer: ' + str(len(bias_all_layer)))
        scale_factor = self.NN.scale_factor

        activation_all_layer = self.NN.activations

        layer_info_all_layer = []
        # initialization for the input layer
        layer_info = []
        for i in range(self.NN.num_of_inputs):
            first_order_der_value = [0]*self.NN.num_of_inputs
            first_order_der_value[i] = 1
            first_order_der_range = [[0,0]]*self.NN.num_of_inputs
            first_order_der_range[i] = [1,1]
            second_order_der_range = [[0,0]]*self.NN.num_of_inputs

            neuron = NeuronInfo(nn_input_dim=self.NN.num_of_inputs, activation_type='Affine', input_value=center[i],
                                input_range=network_input_box[i], first_order_der_value=first_order_der_value,
                                first_order_der_range=first_order_der_range,
                                second_order_der_range=second_order_der_range)
            neuron.set_activation_info()

            layer_info.append(neuron)

        layer_info_all_layer.append(layer_info)

        last_layer_info = layer_info
        for s in range(self.NN.num_of_hidden_layers+1):
            print('layer: ' + str(s))
            this_layer_info = []
            weight = weight_all_layer[s]
            bias = bias_all_layer[s]
            for i in range(self.NN.network_structure[s]):
                print('i: '+str(i))
                neuron = NeuronInfo(nn_input_dim=self.NN.num_of_inputs, activation_type=activation_all_layer[s],
                                    input_value=None,
                                    input_range=None,
                                    first_order_der_value=[],
                                    first_order_der_range=[],
                                    second_order_der_range=[])
                # print('weight_i: ' + str(weight[i,:]))
                # print('bias_i: ' + str(bias[i]))
                neuron.set_input_value(last_layer_info, weight[i,:], bias[i])
                neuron.set_input_range(last_layer_info, weight[i,:], bias[i])
                neuron.set_first_order_der_value(last_layer_info,weight[i])
                neuron.set_first_order_der_range(last_layer_info,weight[i])
                neuron.set_second_order_der_range(last_layer_info,weight[i])
                neuron.set_activation_info()
                this_layer_info.append(neuron)
            layer_info_all_layer.append(this_layer_info)
            last_layer_info = this_layer_info

        output_neuron = layer_info_all_layer[-1][output_index]
        gradient = output_neuron.first_order_der_value
        print('output_der: ' + str(gradient))
        max_hessian = np.abs(np.array(gradient)) + np.multiply(np.array([max(np.abs(np.array(interval))) for interval in output_neuron.second_order_der_range]), np.array(half_len))

        L = LA.norm(max_hessian)
        return L*self.NN.scale_factor

    def get_global_lips(self):
        return self.NN.lips

    def get_output_range_by_local_lips(self, network_input_box, output_index):
        # center
        center = [0.5 * (interval[0] + interval[1]) for interval in network_input_box]
        half_len = [0.5 * (interval[1] - interval[0]) for interval in network_input_box]
        center_output = self.NN.controller(np.array(center))

        local_lips = self.get_local_lips(network_input_box, output_index)
        output_range = [center_output[output_index]-local_lips*half_len[output_index], center_output[output_index]+local_lips*half_len[output_index]]
        return output_range

    def get_output_range_by_global_lips(self, network_input_box, output_index):
        # center
        center = [0.5 * (interval[0] + interval[1]) for interval in network_input_box]
        half_len = [0.5 * (interval[1] - interval[0]) for interval in network_input_box]
        center_output = self.NN.controller(np.array(center))

        global_lips = self.NN.lips()
        output_range = [center_output[output_index]-local_lips*half_len[output_index], center_output[output_index]+global_lips*half_len[output_index]]
        return output_range

    # another representation
    def get_output_center(self, network_input_box, output_index):
        center = [0.5 * (interval[0] + interval[1]) for interval in network_input_box]
        center_output = self.NN.controller(np.array(center))
        return str(center_output[output_index])

    def get_output_error(self, network_input_box, output_index):
        half_len = [0.5 * (interval[1] - interval[0]) for interval in network_input_box]
        local_lips = self.get_local_lips(network_input_box, output_index)
        return str(local_lips*half_len[output_index])