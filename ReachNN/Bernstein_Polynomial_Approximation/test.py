import error_analysis as ea
import sympy as sp
import ast
from network_parser import nn_controller_details
from nn_analysis import NNRange

nn = nn_controller_details('nn_1_sigmoid', 'sigmoid', reuse=True)
box = [[0.7,0.9],[0.7,0.9]]
output_i = 0
nn_range = NNRange(nn)

print('center: ' + str(nn_range.get_output_center(box, output_i)))
print('local_lips: ' + str(nn_range.get_local_lips(box, output_i)))
print('global_lips: ' + str(nn_range.get_global_lips()))