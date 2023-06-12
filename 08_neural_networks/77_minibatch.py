
import time
import load_tensor

import torch
from torch import nn
from torch.utils.data import DataLoader

from utils import TextDataset
from models import NetSingleLayer


train_X, train_Y = load_tensor.load_tensor("train")
valid_X, valid_Y = load_tensor.load_tensor("valid")

net = NetSingleLayer(in_shape=train_X.shape[1], out_shape=4)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

dataset = TextDataset(train_X, train_Y)

batchsizes = [1, 2, 4, 8, 16, 32, 64, 128]
for batchsize in batchsizes:
    loader = DataLoader(dataset, batch_size=batchsize, shuffle=True)

    for epoch in range(1):

        start = time.time()

        # learn weight per batchsize
        for dataloader_x, dataloader_y in loader:

            optimizer.zero_grad()

            dataloader_y_pred_prob = net(dataloader_x)

            dataloader_loss = loss(dataloader_y_pred_prob, dataloader_y)
            dataloader_loss.backward()

            optimizer.step()

        calcuration_time = time.time() - start
        print(f"batchsize{batchsize} time:{calcuration_time: .2f}")

# batchsize1 time: 129.30
# batchsize2 time: 87.99
# batchsize4 time: 71.41
# batchsize8 time: 63.51
# batchsize16 time: 56.37
# batchsize32 time: 55.99
# batchsize64 time: 54.05
# batchsize128 time: 51.74
