import numpy as np
import torch
import os
x1 = np.random.uniform(-10, 10, [1024,4,3]).astype(np.float32)
x2 = np.random.uniform(-10, 10, [1,4,3]).astype(np.float32)
x1 = torch.tensor(x1)
x2 = torch.tensor(x2)
dim = 2
res = torch.cross(x1,x2,dim=dim)
x1 = x1.numpy().astype(np.float32).flatten()
x2 = x2.numpy().astype(np.float32).flatten()
res = res.numpy().astype(np.float32).flatten()
with open("x1.txt", "w") as fp:
    for i in range(len(x1)):
        print(float(x1[i]), file=fp)
with open("x2.txt", "w") as fp:
    for i in range(len(x2)):
        print(float(x2[i]), file=fp)
with open("res.txt", "w") as fp:
    for i in range(len(res)):
        print(float(res[i]), file=fp)