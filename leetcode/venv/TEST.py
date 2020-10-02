import numpy as np
a = [1, 2, 3, 4, 5]
a_mean = sum(a)/len(a)
print(np.transpose(a)- a_mean)