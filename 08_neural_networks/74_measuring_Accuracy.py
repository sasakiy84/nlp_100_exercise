import load_tensor
import torch
from torch import nn


# modelの設定
net = nn.Linear(300, 4)

net_path = "73_net.pth"
net.load_state_dict(torch.load(net_path))


train_X, train_Y = load_tensor.load_tensor("train")
test_X, test_Y = load_tensor.load_tensor("test")


# 学習データに対する予測
train_pred_prob = net(train_X)
_, train_pred = torch.max(train_pred_prob, 1)

# 学習データに対する正解率の計算
train_correct_num = (train_pred == train_Y).sum().item()
train_size = train_Y.size(0)
train_acc = (train_correct_num / train_size) * 100
print(f"train acc:{train_acc: .2f}%")

# 評価データに対する予測
test_pred_prob = net(test_X)
_, test_pred = torch.max(test_pred_prob, 1)

# 評価データに対する正解率の計算
test_correct_num = (test_pred == test_Y).sum().item()
test_size = test_Y.size(0)
test_acc = (test_correct_num / test_size) * 100
print(f"test acc:{test_acc: .2f}%")
# train acc: 57.89%
# test acc: 58.85%
