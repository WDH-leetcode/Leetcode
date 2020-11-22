import matplotlib.pyplot as plt
import numpy as np
import math
x_labeled = [[1, 1], [2, 1], [4, 2]]
x_u = 3
u1 = [1.5, 1.7227]
u2 = [4, 3.6055]
sigma1 = [1]
sigma2 = [1]
pi1 = 0.5
pi2 = 0.5
t = 0
# when difference between theta of current iteration is very close to the last iteration, stop
# keep track of theta and the responsibility
gamma1 = [0.3487]
gamma2 = [0.6515]
while abs(u1[-1] - u1[-2]) >= 0.0001 or abs(u2[-1] - u2[-2]) >= 0.0001:
#while t <= 10:
    alpha = (0.5/np.sqrt(2*3.1415))*(np.exp(-(3-u1[-1])**2/2) + np.exp(-(3-u2[-1])**2/2))
    new_gamma1 = (0.5/(alpha * np.sqrt(2*3.1415)))* math.exp(-(x_u-u1[-1])**2/2)
    new_gamma2 = (0.5/(alpha * np.sqrt(2*3.1415)))* math.exp(-(x_u-u2[-1])**2/2)
    # append new gamma
    gamma1.append(new_gamma1)
    gamma2.append(new_gamma2)
    # update new  theta
    new_u1 = (2*new_gamma1*3 +6)/(2*(new_gamma1+2))
    new_u2 = (2*new_gamma2*3+8)/(2*(new_gamma2+1))
    u1.append(new_u1)
    u2.append(new_u2)
    t += 1

    print(new_gamma1, new_gamma2, new_u1, new_u2)
# compensate for the iteration done by hand
t += 1
x = []
for i in range(t+1):
    x.append(i+1)
# plot
plt.figure(1)
plt.plot(x, u1)
plt.plot(x, u2)
plt.legend(('mu1', 'mu2'))
plt.figure(2)
plt.plot(x[:-1], gamma1)
plt.plot(x[:-1], gamma2)
plt.legend(('gamma1', 'gamma2'))
plt.show()
