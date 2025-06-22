import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, TensorDataset

def train(model: nn.Module, X, y,
          train_ratio: float, lr: float,
          batch_size: int, epochs: int):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    data = TensorDataset(torch.tensor(X, dtype=torch.float32),
                         torch.tensor(y, dtype=torch.float32))
    n_train = int(len(data) * train_ratio)
    train_set, val_set = random_split(data, [n_train, len(data) - n_train])
    loader_tr = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(val_set, batch_size=batch_size)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    history = {'train_loss': [], 'val_loss': []}
    for ep in range(epochs):
        model.train(); tl=0
        for xb, yb in loader_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb).squeeze(), yb)
            loss.backward(); opt.step()
            tl += loss.item()
        model.eval(); vl=0
        with torch.no_grad():
            for xb, yb in loader_val:
                xb, yb = xb.to(device), yb.to(device)
                vl += crit(model(xb).squeeze(), yb).item()
        history['train_loss'].append(tl/len(train_set))
        history['val_loss'].append(vl/len(val_set))
    return history
