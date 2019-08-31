import torch as t
import math
import numpy as np

alist = t.randn(2, 3, 9)

inputsz = np.array(alist.shape[2:])
outputsz = np.array([9])

stridesz = np.floor(inputsz / outputsz).astype(np.int32)
print("stridesz",stridesz)
kernelsz = inputsz - (outputsz - 1) * stridesz
print("kernelsz",kernelsz)

adp = t.nn.AdaptiveMaxPool1d([10])
avg = t.nn.MaxPool1d(kernel_size=list(kernelsz), stride=list(stridesz))
adplist = adp(alist)
avglist = avg(alist)

print(alist)
print(adplist)
print(avglist)
