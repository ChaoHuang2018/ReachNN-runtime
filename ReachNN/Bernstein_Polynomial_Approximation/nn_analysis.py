import numpy as np
import tensorflow as tf
import itertools
import math
import random
import time
import copy
from neruon import NeuronInfo
from numpy import linalg as LA
from interval import interval, inf, imath

class NNTaylor(object):

    def __init__(
        self,
        NN
    ):
        # neural networks
        self.NN = NN
        print(self.NN.num_of_inputs)
        print(self.NN.activations)
        print('self.NN.network_structure' + str(self.NN.network_structure))

    def get_taylor_linear(self, state_vars, network_input_box, output_index):
        # center
        center = [0.5 * (interval_bound[0] + interval_bound[1]) for interval_bound in network_input_box]
        half_len = [0.5 * (interval_bound[1] - interval_bound[0]) for interval_bound in network_input_box]
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            gradient = self.NN.get_gradient(sess, np.array(center).reshape(1, self.NN.num_of_inputs))
            gradient = gradient[0].reshape(self.NN.num_of_inputs)
            print(gradient)
        linear_taylor = self.NN.controller(np.array(center))
        print('NN output on center: ' + str(linear_taylor))
        for i in range(len(gradient)):
            linear_taylor = linear_taylor + gradient[i] * (state_vars[i]-center[i])
        return linear_taylor[0,0]

    def get_taylor_remainder(self, network_input_box, output_index):
        # center
        center = [0.5 * (interval_bound[0] + interval_bound[1]) for interval_bound in network_input_box]
        half_len = [0.5 * (interval_bound[1] - interval_bound[0]) for interval_bound in network_input_box]

        weight_all_layer = self.NN.weights
        bias_all_layer = self.NN.bias
        scale_factor = self.NN.scale_factor

        activation_all_layer = self.NN.activations

        layer_info_all_layer = []
        # initialization for the input layer
        layer_info = []
        for i in range(self.NN.num_of_inputs):
            first_order_der_value = np.zeros(self.NN.num_of_inputs)
            first_order_der_value[i] = 1
            first_order_der_low = np.full(self.NN.num_of_inputs, 1)
            first_order_der_upp = np.full(self.NN.num_of_inputs, 1)

            second_order_der_value = np.zeros((self.NN.num_of_inputs, self.NN.num_of_inputs))
            second_order_der_low = np.zeros((self.NN.num_of_inputs, self.NN.num_of_inputs))
            second_order_der_upp = np.zeros((self.NN.num_of_inputs, self.NN.num_of_inputs))

            neuron = NeuronInfo(nn_input_dim=self.NN.num_of_inputs, activation_type='Affine', input_value=center[i],
                                input_range=interval(network_input_box[i]), first_order_der_value=first_order_der_value,
                                first_order_der_low=first_order_der_low, first_order_der_upp=first_order_der_upp,
                                second_order_der_value=second_order_der_value, second_order_der_low=second_order_der_low,
                                second_order_der_upp=second_order_der_upp)
            neuron.set_activation_info()

            layer_info.append(neuron)

        layer_info_all_layer.append(layer_info)

        last_layer_info = layer_info
        for s in range(self.NN.num_of_hidden_layers + 1):
            this_layer_info = []
            weight = weight_all_layer[s]
            bias = bias_all_layer[s]
            for i in range(self.NN.network_structure[s]):
                neuron = NeuronInfo(nn_input_dim=self.NN.num_of_inputs, activation_type=activation_all_layer[s])
                # print('weight_i: ' + str(weight[i,:]))
                # print(s)
                # print('bias_i: ' + str(bias[i]))
                neuron.set_input_value(last_layer_info, weight[i, :], bias[i])
                neuron.set_input_range(last_layer_info, weight[i, :], bias[i])
                neuron.set_first_order_der_value(last_layer_info, weight[i])
                neuron.set_first_order_der_range(last_layer_info, weight[i])
                neuron.set_second_order_der_value(last_layer_info, weight[i])
                neuron.set_second_order_der_range(last_layer_info, weight[i])
                neuron.set_activation_info()
                this_layer_info.append(neuron)
            layer_info_all_layer.append(this_layer_info)
            last_layer_info = this_layer_info

        output_neuron = layer_info_all_layer[-1][output_index]
        # construct a virtual neruon to represnt the neural network output
        virtual_out_neuron = NeuronInfo(nn_input_dim=self.NN.num_of_inputs, activation_type='Affine')
        virtual_out_neuron.set_input_value(last_layer_info, [self.NN.scale_factor], [-self.NN.offset*self.NN.scale_factor])
        virtual_out_neuron.set_input_range(last_layer_info, [self.NN.scale_factor], [-self.NN.offset*self.NN.scale_factor])
        virtual_out_neuron.set_first_order_der_value(last_layer_info, [self.NN.scale_factor])
        virtual_out_neuron.set_first_order_der_range(last_layer_info, [self.NN.scale_factor])
        virtual_out_neuron.set_second_order_der_value(last_layer_info, [self.NN.scale_factor])
        virtual_out_neuron.set_second_order_der_range(last_layer_info, [self.NN.scale_factor])
        virtual_out_neuron.set_activation_info()

        gradient = virtual_out_neuron.first_order_der_value
        print('output_der on center: ' + str(gradient))
        print('output by our approach: ' + str(virtual_out_neuron.activation_info.value))

        print('output range by IBP : ' + str(virtual_out_neuron.activation_info.output_range))
        print('output range by lips: ' + str(self.NN.lips * LA.norm(np.array(half_len), 2)))

        hessian = virtual_out_neuron.second_order_der_value
        hessian_low = virtual_out_neuron.second_order_der_low
        hessian_upp = virtual_out_neuron.second_order_der_upp

        hessian_max = np.zeros((self.NN.num_of_inputs,self.NN.num_of_inputs))
        for i in range(self.NN.num_of_inputs):
            for j in range(self.NN.num_of_inputs):
                hessian_max[i,j] = max(np.absolute(hessian_low[i,j]), np.absolute(hessian_upp[i,j]))

        error = 0.5 * LA.norm(np.array(half_len), np.inf) * LA.norm(hessian_max, np.inf)
        return error

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
        # construct a virtual neruon to represnt the neural network output
        virtual_out_neuron = NeuronInfo(nn_input_dim=self.NN.num_of_inputs, activation_type='Affine',
                                        input_value=None,
                                        input_range=None,
                                        first_order_der_value=[],
                                        first_order_der_range=[],
                                        second_order_der_range=[])
        virtual_out_neuron.set_input_value(last_layer_info, [self.NN.scale_factor], [self.NN.offset])
        virtual_out_neuron.set_input_range(last_layer_info, [self.NN.scale_factor], [self.NN.offset])
        virtual_out_neuron.set_first_order_der_value(last_layer_info, [self.NN.scale_factor])
        virtual_out_neuron.set_first_order_der_range(last_layer_info, [self.NN.scale_factor])
        virtual_out_neuron.set_second_order_der_range(last_layer_info, [self.NN.scale_factor])
        virtual_out_neuron.set_activation_info()
        gradient = virtual_out_neuron.first_order_der_value

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            #print('call: ' + str(self.NN(sess, np.array(center).reshape(1,2))))
            print('output_der on center by tensorflow: ' + str(self.NN.get_gradient(sess, np.array(center).reshape(1,2))))
        print('output_der on center: ' + str(gradient))
        max_hessian = np.abs(np.array(gradient)) + np.multiply(np.array([max(np.abs(np.array(interval))) for interval in output_neuron.second_order_der_range]), np.array(half_len))

        L = LA.norm(max_hessian)
        return L

    def get_global_lips(self):
        return self.NN.lips

    def get_output_range_by_local_lips(self, network_input_box, output_index, partition):
        partition = [partition] * self.NN.num_of_inputs
        point_index_list = degree_comb_lists(partition, self.NN.num_of_inputs)
        distance = 0
        for point_index in point_index_list:
            point = [0] * self.NN.num_of_inputs
            for i in range(self.NN.num_of_inputs):
                side_len = (network_input_box[i][1] - network_input_box[i][0])/partition[0]
                point[i] = side_len * (point_index[i] + 0.5) + network_input_box[i][0]
                distance = side_len * side_len
        distance = math.sqrt(distance)
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
        output_range = [center_output[output_index]-global_lips*half_len[output_index], center_output[output_index]+global_lips*half_len[output_index]]
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


def degree_comb_lists(d, m):
    # generate the degree combination list
    if m == 1:
        X = np.meshgrid(np.arange(d[0]))
        return [tuple(row) for row in X]
    if m == 2:
        x = np.arange(d[0])
        y = np.arange(d[1])
        X, Y = np.meshgrid(x, y)
        grid = np.vstack((X.flatten(), Y.flatten()))
        return grid.T
    if m == 3:
        x = np.arange(d[0])
        y = np.arange(d[1])
        z = np.arange(d[2])
        X, Y, Z = np.meshgrid(x, y, z)
        grid = np.vstack((X.flatten(), Y.flatten(), Z.flatten()))
        return grid.T
    if m == 4:
        x = np.arange(d[0])
        y = np.arange(d[1])
        z = np.arange(d[2])
        h = np.arange(d[3])
        X, Y, Z, H = np.meshgrid(x, y, z, h)
        grid = np.vstack((X.flatten(), Y.flatten(), Z.flatten(), H.flatten()))
        return grid.T

