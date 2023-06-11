import load_tensor
import torch
from torch import nn
from utils import make_graph


def calc_acc(y_pred_prob, y_true) -> float:
    """予測のtensorの正解のtensorを用いて、正解率を計算する"""
    # 最も正解率の高い予測確率を正解ラベルとする。
    _, y_pred = torch.max(y_pred_prob, 1)

    # 学習データに対する正解率の計算
    correct_num = (y_pred == y_true).sum().item()
    total_size = y_true.size(0)
    acc = (correct_num / total_size) * 100
    return acc


train_X, train_Y = load_tensor.load_tensor("train")
valid_X, valid_Y = load_tensor.load_tensor("valid")

# modelの設定
net = nn.Linear(300, 4)

# loss, optimizerの設定
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

train_losses = []
train_accs = []
valid_losses = []
valid_accs = []

# parameterの更新
for epoch in range(100):
    optimizer.zero_grad()

    train_y_pred_prob = net(train_X)

    # 訓練データでの損失の計算
    train_loss = loss(train_y_pred_prob, train_Y)
    train_loss.backward()

    optimizer.step()

    # 訓練データでの損失の保存
    train_losses.append(train_loss.data)

    # 訓練データでの正解率の計算
    train_acc = calc_acc(train_y_pred_prob, train_Y)
    # 訓練データでの正解率の保存
    train_accs.append(train_acc)

    # 検証データに対する予測
    valid_y_pred_prob = net(valid_X)

    # 検証データの損失の計算
    valid_loss = loss(valid_y_pred_prob, valid_Y)
    # 検証データでの損失の保存
    valid_losses.append(valid_loss.data)

    # 検証データでの正解率の計算
    valid_acc = calc_acc(valid_y_pred_prob, valid_Y)
    # 検証データでの正解率の保存
    valid_accs.append(valid_acc)

# グラフへのプロット
losses = {"train": train_losses, "valid": valid_losses}

accs = {"train": train_accs, "valid": valid_accs}

make_graph(losses, "losses", 75)
make_graph(accs, "accs", 75)
