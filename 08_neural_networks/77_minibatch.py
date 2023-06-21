
import time
import load_tensor

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from utils import TextDataset, calc_acc, make_graph
from models import NetSingleLayer


train_X, train_Y = load_tensor.load_tensor("train")
valid_X, valid_Y = load_tensor.load_tensor("valid")


dataset = TextDataset(train_X, train_Y)
valid_losses = []
valid_accs = []
train_losses = []
train_accs = []

# we can decrease an amount of memory usage when batch size is small
batchsizes = [1]
for batchsize in batchsizes:
    net = nn.Linear(300, 4)

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    loader = DataLoader(dataset, batch_size=batchsize, shuffle=True)

    for epoch in range(100):

        start = time.time()

        # learn weight per batchsize
        for dataloader_x, dataloader_y in loader:

            optimizer.zero_grad()

            dataloader_y_pred_prob = net(dataloader_x)

            dataloader_loss: Tensor = loss(
                dataloader_y_pred_prob, dataloader_y)
            dataloader_loss.backward()

            optimizer.step()

        calcuration_time = time.time() - start
        # print(f"batchsize{batchsize} time:{calcuration_time: .2f}")

        train_y_pred_prob = net(train_X)
        train_acc = calc_acc(train_y_pred_prob, train_Y)

        valid_y_pred_prob = net(valid_X)
        valid_acc = calc_acc(valid_y_pred_prob, valid_Y)
        valid_loss = loss(valid_y_pred_prob, valid_Y)
        train_loss = loss(train_y_pred_prob, train_Y)

        print(f"{batchsize}: acc={valid_acc} loss={valid_loss}")
        train_accs.append(train_acc)
        train_losses.append(train_loss)
        valid_accs.append(valid_acc)
        valid_losses.append(valid_loss)

losses = {"train": train_losses, "valid": valid_losses}
accs = {"train": train_accs, "valid": valid_accs}
make_graph(losses, "losses", 77)
make_graph(accs, "accs", 77)

# for i in range(len(batchsizes)):
#     print(f"{batchsizes[i]}: acc={valid_accs[i]} loss={valid_losses[i]}")


# batchsize1 time: 129.30
# batchsize2 time: 87.99
# batchsize4 time: 71.41
# batchsize8 time: 63.51
# batchsize16 time: 56.37
# batchsize32 time: 55.99
# batchsize64 time: 54.05
# batchsize128 time: 51.74

# 1: acc=77.97752380371094 loss=1.0217046737670898
# 2: acc=77.15355682373047 loss=1.0698373317718506
# 4: acc=77.22846221923828 loss=1.1494355201721191
# 8: acc=77.07865142822266 loss=1.2339171171188354
# 16: acc=73.40824127197266 loss=1.303740382194519
# 32: acc=74.30712127685547 loss=1.3409559726715088
# 64: acc=47.64044952392578 loss=1.367052674293518
# 128: acc=46.81647872924805 loss=1.374577522277832
