import pickle
import numpy as np
import pandas as pd
import torch
from torch import nn

train: pd.DataFrame = pickle.load(open("train.pickle", "rb"))

X_df = train["TITLE_VECTOR"]

# convert data_series of numpy._object to pure numpy matrix
X = []
for row in X_df:
    X.append(row.tolist())
X = np.array(X)

X = torch.tensor(X, requires_grad=True).float()
W = torch.randn(300, 4, dtype=torch.float)
# expect (10685, 300) * (300, 4) = (10685, 4)
print(X.shape, W.shape)
XW = torch.matmul(X, W)
m = nn.Softmax(dim=1)
output = m(XW)

print(output[0])
print(torch.sum(output, 1))
torch.save(output, "71.pt")
