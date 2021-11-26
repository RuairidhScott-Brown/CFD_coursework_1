import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import toeplitz
import scipy
import pprint

number_of_data_points = 7
A = np.zeros(((number_of_data_points-2)**2, (number_of_data_points-2)**2))
A_shape = A.shape
four = np.ones((number_of_data_points-2)**2)*-4
ones = np.ones((number_of_data_points-2)**2-1)
ones_2 = np.ones((number_of_data_points-2)**2 - (number_of_data_points-2))
for i in range(len(ones)):
    if (i+1) % (number_of_data_points - 2) == 0:
        ones[i] =0

test = np.diag(four)
test2 = np.diag(ones, 1)
test3 = np.diag(ones, -1)
test4 = np.diag(ones_2, number_of_data_points-2)
test5 = np.diag(ones_2, -number_of_data_points+2)
Full_matrix = test+test2+test3+test4+test5
print(Full_matrix)
np.savetxt("A.txt", Full_matrix)
print(f"Shape of the A matrix: {A_shape}")