import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, TensorDataset

class Trainer:
    """
    Train PyTorch model, record history (loss, metrics).
    """
    def __init__(self, model: nn.Module, X, y, train_ratio: float, lr: float, batch_size: int = 32):
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.lr = lr
        self.batch_size = batch_size
        # prepare dataset
        data = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
        n = len(data)
        n_train = int(train_ratio * n)
        self.train_set, self.val_set = random_split(data, [n_train, n - n_train])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.history = {'train_loss': [], 'val_loss': []}

    def train(self, epochs: int = 50):
        for ep in range(epochs):
            # train
            self.model.train()
            tl = 0
            for xb, yb in DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True):
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                pred = self.model(xb).squeeze()
                loss = self.criterion(pred, yb)
                loss.backward()
                self.optimizer.step()
                tl += loss.item()
            # val
            self.model.eval()
            vl = 0
            with torch.no_grad():
                for xb, yb in DataLoader(self.val_set, batch_size=self.batch_size):
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    pred = self.model(xb).squeeze()
                    vl += self.criterion(pred, yb).item()
            self.history['train_loss'].append(tl/len(self.train_set))
            self.history['val_loss'].append(vl/len(self.val_set))
        return self.history
