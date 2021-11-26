import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import toeplitz
import scipy
import pprint





number_of_data_points = 11
middle = ((number_of_data_points - 1)/2)
quater = ((number_of_data_points-1)/4)

delta_x = 1/number_of_data_points
delta_y = 1/number_of_data_points

x = np.linspace(0, 1, number_of_data_points)
y = np.linspace(0, 1, number_of_data_points)

def BC_x0(y):
    return 1

def BC_x1(y):
    return np.cos(6 * (2/3) * np.pi * y) + 1

def BC_y0(x):
    return 1 + x

def BC_y1(x):
    return 1

xx, yy = np.meshgrid(x, y, sparse=False)


T = np.zeros((number_of_data_points,number_of_data_points))

# Setting up BC
T[0] = np.ones((1,number_of_data_points))

for row in T:
    row[0] = 1

x_with_BC_y0 = xx + 1

y_with_BC_x1 = np.cos(6 * (3/2) * np.pi * yy) + 1

T[-1] = x_with_BC_y0[0]

counter = -1
for row in T:
    row[-1] = y_with_BC_x1[counter][0]
    counter -= 1

T[int(middle)][int(middle)] = 2.5
T[-int(quater)][int(quater)] = -0.5

#print(T)
np.savetxt("Temperature.txt", T)
temp1 = np.zeros(number_of_data_points**2)
temp1_1 = temp1
temp1_1[0] = -4
temp1_1[1] = 1
temp1_1[4] = 1
first_row = np.array(temp1_1)
first_column = np.array(temp1_1)
diag = toeplitz(first_column, first_row)
#pprint.pprint(diag)


# Creating the Matric to multiply
A = np.zeros(((number_of_data_points-2)**2, (number_of_data_points-2)**2))
A_shape = A.shape
four = np.ones((number_of_data_points-2)**2)*-4
test = np.diag(four)
test2 = np.diag(four, 1)
print(test)
print(test2)
print(f"Shape of the A matrix: {A_shape}")


#for index in range(number_of_data_points-2):

D1 = T.reshape((1,number_of_data_points**2))

def back_sub(A,b):
    n = len(A)
    x = [0]*n
    for i in range(n-1,-1,-1): #this refers to the rows; i goes 2,1,0
        for j in range(i+1,n): #j goes 1,2 @ i = 0
                               #j goes 2   @ i = 1
            b[i] = b[i] - A[i,j]*x[j]
        x[i] = b[i]/A[i,i]

    return np.array(x)

M = np.array([[1, -1, 2], [0, -1, -2], [0, 0, -6]])
c = np.array([[0],[0],[3]])


P, L, U = scipy.linalg.lu(diag)

