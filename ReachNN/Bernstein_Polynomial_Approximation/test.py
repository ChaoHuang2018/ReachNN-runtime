import error_analysis as ea
import sympy as sp
import ast
from network_parser import nn_controller_details
from nn_analysis import NNTaylor
import time


nn = nn_controller_details('nn_1_sigmoid', 'sigmoid', reuse=True)
box = [[0.4,0.5],[0.4,0.5]]
output_i = 0
nn_range = NNTaylor(nn)

x = sp.symbols('x:'+str(nn.num_of_inputs))

time_start = time.time()
print('Linear Taylor: ' + str(nn_range.get_taylor_linear(x, box, output_i)))
time_expression = time.time() - time_start
print('Time for generating linear taylor: ' + str(time_expression))
time_start = time.time()
print('Error: ' + str(nn_range.get_taylor_remainder(box, output_i)))
time_error = time.time() - time_start
print('Time for generating error: ' + str(time_error))