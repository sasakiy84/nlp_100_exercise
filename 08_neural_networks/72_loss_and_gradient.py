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

# tensor(1.3668, grad_fn=<NllLossBackward0>)
# tensor([[-6.5584e-05,  1.9824e-05,  2.4159e-05,  2.1602e-05],
#         [ 1.9099e-05, -5.8209e-05,  2.0320e-05,  1.8790e-05],
#         [ 1.8258e-05,  3.2146e-05,  2.3402e-05, -7.3805e-05],
#         ...,
#         [ 2.1854e-05,  2.1254e-05, -6.5475e-05,  2.2367e-05],
#         [ 2.9607e-05,  2.6054e-05, -7.5438e-05,  1.9778e-05],
#         [-7.2888e-05,  2.7122e-05,  2.6973e-05,  1.8792e-05]])
