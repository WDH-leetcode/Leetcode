import numpy as np
import matplotlib.pyplot as plt
xt = np.genfromtxt('/Users/huangweiding/Documents/leetcode/EE660/hw2_data/x_train.csv', delimiter=',')
yt = np.genfromtxt('/Users/huangweiding/Documents/leetcode/EE660/hw2_data/y_train.csv', delimiter=',')
# c
plt.scatter(xt, yt)
#plt.show()

#d
xt = np.reshape(xt, (25, 1))
yt = np.reshape(yt, (25, 1))


def fib(n, x):
    ones = np.ones((len(x), 1))
    if n == 1:
        arr = np.concatenate((ones, x), axis=1)
    else:
        prev = fib(n-1, x)

        arr = np.concatenate((prev, np.power(x, n)), axis=1)
    return arr


arr1 = fib(1, xt)
arr2 = fib(2, xt)
arr3 = fib(3, xt)
arr7 = fib(7, xt)
arr10 = fib(10, xt)

# pesudoinverse w = (x' * x)^(-1) * x' * y


def wpinv(arr):
    w = np.dot(np.linalg.pinv(arr), yt)
    return w


w1 = wpinv(arr1)
w2 = wpinv(arr2)
w3 = wpinv(arr3)
w7 = wpinv(arr7)
w10 = wpinv(arr10)
#print(w1)
#print(w2)
#print(w3)

#print((yt - np.dot(arr1, w1)))
# e


def mse(w, arr, y):
    square = (y - np.dot(arr, w))
    d = 0
    for i in square:
        d += i**2
    return d/len(arr)


d_array = [mse(w1, arr1, yt), mse(w2, arr2, yt), mse(w3, arr3, yt), mse(w7, arr7, yt), mse(w10, arr10, yt)]
x_label = [1, 2, 3, 7, 10]

plt.figure(2)
plt.plot(x_label, d_array)
#plt.show()

# f

x_test = np.genfromtxt('/Users/huangweiding/Documents/leetcode/EE660/hw2_data/x_test.csv', delimiter=',')
y_test = np.genfromtxt('/Users/huangweiding/Documents/leetcode/EE660/hw2_data/y_test.csv', delimiter=',')
x_test = np.reshape(x_test, (len(x_test), 1))
y_test = np.reshape(y_test, (len(y_test), 1))

arr1_test = fib(1, x_test)
arr2_test = fib(2, x_test)
arr3_test = fib(3, x_test)
arr7_test = fib(7, x_test)
arr10_test = fib(10, x_test)
d_array_test = [mse(w1, arr1_test, y_test), mse(w2, arr2_test, y_test), mse(w3, arr3_test, y_test), mse(w7, arr7_test, y_test), mse(w10, arr10_test, y_test)]
plt.figure(3)
plt.plot(x_label, d_array_test)
#plt.show()

# g
lam = [10**(-5), 10**(-3), 10**(-1), 1, 10]
I = np.identity(8)
w_list = list()
for i in lam:
    w = np.dot(np.dot(np.linalg.inv(np.dot(i, I) + np.dot(np.transpose(arr7), arr7)), np.transpose(arr7)), yt)
    w_list.append(w)

# h

MSE_train = [mse(w_list[0], arr7, yt), mse(w_list[1], arr7, yt), mse(w_list[2], arr7, yt), mse(w_list[3], arr7, yt), mse(w_list[4], arr7, yt)]
MSE_test = [mse(w_list[0], arr7_test, y_test), mse(w_list[1], arr7_test, y_test), mse(w_list[2], arr7_test, y_test), mse(w_list[3], arr7_test, y_test), mse(w_list[4], arr7_test, y_test)]
log_lam = np.log(lam)
MSE_train = np.log(MSE_train)
MSE_test = np.log(MSE_test)
plt.figure(4)
plt.plot(log_lam, MSE_train)
plt.plot(log_lam, MSE_test)
plt.show()
