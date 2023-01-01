import torch.nn as nn
import torch

lossFn = nn.CrossEntropyLoss()

pred = torch.tensor([[5., 7.]])
p = torch.softmax(pred, dim=1)
print(p)
y_true = torch.tensor([1])

print(lossFn(pred, y_true))

print( - torch.log(p[0,1] * 1))

