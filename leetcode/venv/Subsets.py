import numpy as np
n = [1, 2, 3, 4, 5]
i = [[]]
for o in n:
    i += [[cur] + [o] for cur in n]
print(i)