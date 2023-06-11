import pandas as pd
import torch
from torch import nn
import load_tensor

X = torch.load("71.pt")

_, Y = load_tensor.load_tensor("train")

# calculate loss
loss = nn.CrossEntropyLoss()
output = loss(X, Y)

print(output)

# calculate gradient
# X is initialized with `required_grad = True`, so gradient of X is caluculated under the hood.
output.backward()
print(X.grad)
