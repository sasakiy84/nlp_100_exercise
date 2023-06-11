from load_tensor import load_tensor
import torch
from torch import nn


X, Y = load_tensor("train")

net = nn.Linear(300, 4)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

print("Before")
print(net.state_dict()["weight"])

for step in range(100):
    optimizer.zero_grad()

    y_pred = net(X)

    output = loss(y_pred, Y)
    output.backward()

    optimizer.step()

print("After")
print(net.state_dict()["weight"])

net_path = "73_net.pth"
torch.save(net.state_dict(), net_path)
