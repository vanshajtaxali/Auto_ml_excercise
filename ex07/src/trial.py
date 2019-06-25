import numpy as np
import torch

list_of_tensors = []

ts = torch.tensor([[10,10,10],[10,10,10]])
print('the first tensor\n',ts)
print('shape',ts.shape)
print(' ')
list_of_tensors.append(ts)

ts_2 = torch.tensor([[10,10,10],[10,10,10]])
ts_3 = torch.tensor([[10,10,10],[10,10,10]])
ts_4 = torch.tensor([[10,10,10],[10,10,10]])
list_of_tensors.append(ts_2)
list_of_tensors.append(ts_3)
list_of_tensors.append(ts_4)
print('the first tensor\n',ts_2)
print('shape',ts_2.shape)
print(' ')

print('the len of the tensors',len(list_of_tensors))
addition = sum(list_of_tensors)
print(addition)


