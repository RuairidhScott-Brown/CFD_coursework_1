import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import toeplitz
import scipy
import pprint

number_of_data_points = 11

number_of_data_points = 101
delta_x = 1/(number_of_data_points - 1)
middle = ((number_of_data_points - 1)/2)
quater = ((number_of_data_points-1)/4)

x = np.linspace(0, 1, number_of_data_points)
y = np.linspace(0, 1, number_of_data_points)

xx, yy = np.meshgrid(x, y, sparse=False)

for i in range(yy.shape[0]):
    yy[i] = (yy[i] - 1)*(-1)

print(yy)

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

T[int(middle)][int(middle)] = -2.5
T[(number_of_data_points- 1) - int(quater)][int(quater)] = 0.5

np.savetxt("temp.txt", T)


B = np.array([], dtype=float)

for i in range(number_of_data_points):
    for j in range(number_of_data_points):
        if i != 0 and i != (number_of_data_points-1) and j != 0 and j != (number_of_data_points-1):
            B = np.append(B,T[i][j])

#print(B)
zero_row = []
for index,element in enumerate(B):
    if element != 0:
        zero_row.append(index)
print(zero_row)

for i in range(number_of_data_points):
    for j in range(number_of_data_points):
        if i == 0 or i == (number_of_data_points-1) or j == 0 or j == (number_of_data_points-1):
            continue
        elif i == 1:
            if j == 1:
                B[0] += + T[0][1] + T[1][0]
            elif j == (number_of_data_points - 2):
                B[number_of_data_points - 3] += T[0][number_of_data_points-2] + T[1][number_of_data_points-1]
            else:
                B[j-1] += T[0][j]
        elif i == (number_of_data_points - 2):
            if j == 1:
                B[(number_of_data_points-2)*(number_of_data_points-3)] += T[number_of_data_points-2][0] + T[number_of_data_points-1][1]
            elif j == (number_of_data_points - 2):
                B[(number_of_data_points-2)**2 - 1] += T[number_of_data_points-1][number_of_data_points-2] + T[number_of_data_points-2][number_of_data_points-1]
            else:
                B[(number_of_data_points-2)*(number_of_data_points-3)+ (j-1)] += T[number_of_data_points-1][j]
        else:
            if j == 1:
                B[(i-1)*(number_of_data_points-2)] += T[i][0]
            elif j == (number_of_data_points - 2):

                B[i*(number_of_data_points-2)-1] += T[i][j+1]
#print(B)


A = np.zeros(((number_of_data_points-2)**2, (number_of_data_points-2)**2), dtype=float)
A_shape = A.shape
four = np.ones((number_of_data_points-2)**2, dtype=float)*-4
ones = np.ones((number_of_data_points-2)**2-1, dtype=float)
ones_2 = np.ones((number_of_data_points-2)**2 - (number_of_data_points-2), dtype=float)
for i in range(len(ones)):
    if (i+1) % (number_of_data_points - 2) == 0:
        ones[i] =0

test = np.diag(four)
test2 = np.diag(ones, 1)
test3 = np.diag(ones, -1)
test4 = np.diag(ones_2, number_of_data_points-2)
test5 = np.diag(ones_2, -number_of_data_points+2)
Full_matrix = test+test2+test3+test4+test5

for i in range(Full_matrix.shape[0]):
    if i in zero_row:
        print(i)
        for j in range(Full_matrix.shape[1]):
            if Full_matrix[i][j] == 1:
                Full_matrix[i][j] = 0
        for j in range(Full_matrix.shape[1]):
            if Full_matrix[i][j] == -4:
                Full_matrix[i][j] = 1
        print(Full_matrix[i])
        



def LU_inplace(A): 
    """Compute the LU factorisation of A, using the in-place scheme so
    that the strictly lower triangular components of the array contain
    the strictly lower triangular components of L, and the upper
    triangular components of the array contain the upper triangular
    components of U.

    :param A: an mxm-dimensional numpy array

    """
    Matrix = A.copy()
    m,n = Matrix.shape
    
    for k in range(m-1):
        Matrix[k+1:,k] = Matrix[k+1:,k]/Matrix[k,k]
        Matrix[k+1:,k+1:] = Matrix[k+1:,k+1:] - np.outer(Matrix[k+1:,k], Matrix[k,k+1:])
    return Matrix

# # = LU_inplace(Full_matrix)
# print(3)
# U = np.triu(m)
# L = np.tril(m, k=-1)
# L = L + np.eye(L.shape[0])
B = -B
# x_dot = np.linalg.inv(L)@B
# print(2)
# x_dot = np.linalg.inv(U)@x_dot
x = scipy.linalg.solve(Full_matrix, B)
z =  x.reshape((number_of_data_points-2, number_of_data_points-2))

#z = x_dot.reshape((number_of_data_points-2, number_of_data_points-2))



#plt.contour(xx, yy, z, colors='black')


for i in range(number_of_data_points):
    for j in range(number_of_data_points):
        if i != 0 and i != (number_of_data_points-1) and j != 0 and j != (number_of_data_points-1):
            T[i][j] = z[i-1][j-1]
print(T)

#plt.contour(xx, yy, T, 20,)

fig, ax = plt.subplots(1,1)
CS = ax.contourf(xx, yy, T, cmap = "viridis")
ax.clabel(CS, colors = "k", inline=False, fontsize=10)
ax.set_xlabel('x', fontsize = 20)
ax.set_ylabel('y', fontsize= 20)
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)
plt.colorbar(CS)

fig = plt.figure()
ax = plt.axes(projection='3d')
CV = ax.plot_surface(xx, yy, T, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_xlabel('x', fontsize= 20)
ax.set_ylabel('y', fontsize= 20)
ax.set_zlabel('T', fontsize= 20)
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)
plt.colorbar(CV)




## Gauss Sediel

Lower = np.tril(Full_matrix, k=-1)
Upper = np.triu(Full_matrix, k=1)
Diag = Full_matrix - Lower - Upper



for
plt.show()
