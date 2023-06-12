import load_tensor
import torch
from torch import nn
from utils import calc_acc

# setup model
net = nn.Linear(300, 4)

net_path = "73_net.pth"
net.load_state_dict(torch.load(net_path))


train_X, train_Y = load_tensor.load_tensor("train")
test_X, test_Y = load_tensor.load_tensor("test")


# predict train data
train_pred_prob = net(train_X)
_, train_pred = torch.max(train_pred_prob, 1)

# calc accurary of train predication
train_correct_num = (train_pred == train_Y).sum().item()
train_size = train_Y.size(0)
train_acc = (train_correct_num / train_size) * 100
print(f"train acc:{train_acc: .2f}%")

# predict test data
test_pred_prob = net(test_X)

test_acc = calc_acc(test_pred_prob, test_Y)
print(f"test acc:{test_acc: .2f}%")
# train acc: 57.89%
# test acc: 58.85%
