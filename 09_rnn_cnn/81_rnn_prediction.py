import pickle
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):  # len(Dataset)で返す値を指定
        return len(self.X)

    def __getitem__(self, idx):  # Dataset[index]で返す値を指定
        # Xをintのlistに変換
        X_list = [int(x) for x in self.X[idx].split()]

        # tensorに変換
        inputs = torch.tensor(
            X_list
        )  # .unsqueeze(0) # unsqueezeはtorch.Size([6]) →　torch.Size([1, 6])
        label = torch.tensor(self.y[idx])

        return inputs, label


class RNN(nn.Module):
    def __init__(self, vocab_size, emb_dim=300, hidden_size=50, output_size=4):
        super().__init__()

        self.emb = nn.Embedding(
            vocab_size, emb_dim, padding_idx=0  # 0に変換された文字にベクトルを計算しない
        )

        self.rnn = nn.RNN(
            input_size=emb_dim,
            hidden_size=hidden_size,
            num_layers=1,
            nonlinearity="tanh",
            bias=True,
        )

        self.fc = nn.Linear(
            in_features=hidden_size, out_features=output_size, bias=True
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, h_0=None):
        x = self.emb(x)
        x, h_t = self.rnn(x, h_0)
        x = x[:, -1, :]
        x = self.fc(x)
        x = self.softmax(x)
        return x


# データの読み込み
train = pd.read_pickle("./train_title_id.pkl")

# yの変換
cat_id_dict = {"b": 0, "t": 1, "e": 2, "m": 3}
train["CATEGORY"] = train["CATEGORY"].map(cat_id_dict)

# 辞書の読み込み
with open("./word_id_dict.pkl", "rb") as tf:
    word_id_dict = pickle.load(tf)

n_letters = len(word_id_dict.keys())
n_hidden = 50
n_categories = 4

# modelの定義
model = RNN(n_letters, n_hidden, n_categories)

# datasetの定義
dataset = TextDataset(train["TITLE"], train["CATEGORY"])

# 先頭10個の結果を出力
for i in range(10):
    X = dataset[i][0]
    X = X.unsqueeze(0)
    print(model(x=X))

# tensor([[0.2670, 0.1487, 0.4902, 0.0941]], grad_fn=<SoftmaxBackward0>)
# tensor([[0.2393, 0.2032, 0.4834, 0.0741]], grad_fn=<SoftmaxBackward0>)
# tensor([[0.1363, 0.1494, 0.4774, 0.2369]], grad_fn=<SoftmaxBackward0>)
# tensor([[0.1529, 0.1441, 0.3553, 0.3477]], grad_fn=<SoftmaxBackward0>)
# tensor([[0.3702, 0.0890, 0.3794, 0.1614]], grad_fn=<SoftmaxBackward0>)
# tensor([[0.0797, 0.4167, 0.3297, 0.1739]], grad_fn=<SoftmaxBackward0>)
# tensor([[0.1546, 0.0571, 0.4542, 0.3341]], grad_fn=<SoftmaxBackward0>)
# tensor([[0.2199, 0.2842, 0.2832, 0.2127]], grad_fn=<SoftmaxBackward0>)
# tensor([[0.1722, 0.4761, 0.2465, 0.1052]], grad_fn=<SoftmaxBackward0>)
# tensor([[0.1126, 0.1401, 0.5975, 0.1499]], grad_fn=<SoftmaxBackward0>)
