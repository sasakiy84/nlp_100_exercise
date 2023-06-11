
import time
import load_tensor

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from utils import make_graph


class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class NetSingleLayer(nn.Module):
    def __init__(self, in_shape: int, out_shape: int):
        super().__init__()
        self.fc = nn.Linear(300, 4, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc(x)
        x = self.softmax(x)
        return x


class NetThreeLayer(nn.Module):
    def __init__(self, in_shape: int, out_shape: int):
        super().__init__()
        self.fc1 = nn.Linear(300, 150, bias=True)
        self.dropout1 = nn.Dropout(0.25)
        self.bn1 = nn.BatchNorm1d(150)
        self.fc2 = nn.Linear(150, 150, bias=True)
        self.dropout2 = nn.Dropout(0.25)
        self.bn2 = nn.BatchNorm1d(150)
        self.fc3 = nn.Linear(300, 4, bias=True)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.fc1(x)
        x1 = self.dropout1(x1)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)

        x2 = self.fc2(x1)
        x2 = self.dropout2(x2)
        x2 = self.bn2(x2)
        x2 = self.relu(x2)
        x2 = torch.cat([x1, x2], dim=1)

        x3 = self.fc3(x2)
        x3 = self.softmax(x3)

        return x3


def calc_acc(net, train_x, y_true) -> float:
    """modelと学習データ、正解データを用いて、正解率を計算する"""
    # 最も正解率の高い予測確率を正解ラベルとする。
    _, y_pred = torch.max(net(train_x), 1)

    # 学習データに対する正解率の計算
    correct_num = (y_pred == y_true).sum().item()
    total_size = y_true.size(0)
    acc = (correct_num / total_size) * 100
    return acc


if torch.cuda.is_available():
    print("cuda is available!!")
else:
    print("No cuda")

device = (
    torch.device("cuda:0") if torch.cuda.is_available(
    ) else torch.device("cpu")
)

train_X, train_Y = load_tensor.load_tensor("train")
valid_X, valid_Y = load_tensor.load_tensor("valid")

train_X, train_Y, valid_X, valid_Y = train_X.to(device), train_Y.to(
    device), valid_X.to(device), valid_Y.to(device)

# modelの設定
net = NetSingleLayer(in_shape=train_X.shape[1], out_shape=4).to(device)
# net = NetThreeLayer(in_shape=train_X.shape[1], out_shape=4).to(device)

# loss, optimizerの設定
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

# DataLoaderの構築
dataset = TextDataset(train_X, train_Y)

# parameterの更新
batchsize = 128
loader = DataLoader(dataset, batch_size=batchsize, shuffle=True)

train_losses = []
train_accs = []
valid_losses = []
valid_accs = []

for epoch in range(100):

    start = time.time()

    train_running_loss = 0.0
    valid_running_loss = 0.0

    for dataloader_x, dataloader_y in loader:
        """netの重みの学習をbatchsize単位で行う"""

        optimizer.zero_grad()

        dataloader_y_pred_prob = net(dataloader_x)

        # dataset_xでの損失の計算
        dataloader_loss = loss(dataloader_y_pred_prob, dataloader_y)
        dataloader_loss.backward()

        # 訓練データ、検証データでの損失の平均を計算する
        train_running_loss += dataloader_loss.item()
        valid_running_loss += loss(net(valid_X), valid_Y).item()

        optimizer.step()

    # 訓練データでの損失の保存
    train_losses.append(train_running_loss)

    # 訓練データでの正解率の計算
    train_acc = calc_acc(net, train_X, train_Y)
    # 訓練データでの正解率の保存
    train_accs.append(train_acc)

    # 検証データでの損失の保存
    valid_losses.append(valid_running_loss)

    # 検証データでの正解率の計算
    valid_acc = calc_acc(net, valid_X, valid_Y)
    # 検証データでの正解率の保存
    valid_accs.append(valid_acc)

    # 20epoch毎にチェックポイントを生成
    if epoch % 20 == 0:
        torch.save(net.state_dict(),
                   f"79_net_bs{batchsize}_epoch{epoch}.pth")
        torch.save(
            optimizer.state_dict(),
            f"79_optimizer_bs{batchsize}_epoch{epoch}.pth",
        )

    # 経過した時間を取得
    elapsed_time = time.time() - start
    print(f"{epoch}: valid_acc={valid_acc} train_acc={train_acc} time={elapsed_time: .2f}")


losses = {"train": train_losses, "valid": valid_losses}

accs = {"train": train_accs, "valid": valid_accs}

make_graph(losses, "losses", 79)
make_graph(accs, "accs", 79)
