import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class SnakeNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 5),
        )

    def forward(self, x):
        return self.net(x)


class SnakeModel:
    """Wrapper exposing fit/predict/save/load around a PyTorch network."""

    def __init__(self, input_size, LR=1e-3):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = SnakeNet(input_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=LR)
        self.loss_fn = nn.CrossEntropyLoss()

    def fit(self, X, y, n_epoch=5, batch_size=64):
        self.net.train()
        inputs = torch.as_tensor(np.asarray(X, dtype=np.float32))
        # CrossEntropyLoss expects class indices, so collapse the one-hot labels
        targets = torch.as_tensor(np.argmax(np.asarray(y), axis=1).astype(np.int64))
        # inverse-frequency class weights counter the heavy "no-op" majority
        counts = np.bincount(targets.numpy(), minlength=5).astype(np.float32)
        class_weights = len(targets) / (5.0 * np.maximum(counts, 1.0))
        self.loss_fn = nn.CrossEntropyLoss(
            weight=torch.as_tensor(class_weights).to(self.device))
        loader = DataLoader(TensorDataset(inputs, targets), batch_size=batch_size, shuffle=True)
        print(f"training on {len(targets)} samples in {len(loader)} batches of {batch_size}")
        for epoch in range(n_epoch):
            total_loss, correct, total = 0.0, 0, 0
            progress = tqdm(loader, desc=f"epoch {epoch + 1}/{n_epoch}", unit="batch")
            for batch_X, batch_y in progress:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                logits = self.net(batch_X)
                loss = self.loss_fn(logits, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * batch_X.size(0)
                correct += (logits.argmax(1) == batch_y).sum().item()
                total += batch_X.size(0)
                progress.set_postfix(loss=total_loss / total, acc=correct / total)
            print(f"epoch {epoch + 1}/{n_epoch} - loss: {total_loss / total:.4f} - acc: {correct / total:.4f}")
        return self

    def predict(self, X):
        self.net.eval()
        with torch.no_grad():
            inputs = torch.as_tensor(np.asarray(X, dtype=np.float32)).to(self.device)
            logits = self.net(inputs)
        return logits.cpu().numpy()

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path, map_location=self.device))
        return self


def neural_network_model(input_size, LR=1e-3):
    """
    Purpose
        Build the neural network used to play the snake game
    Input
        input_size, int: size of the flattened grid input
        LR, float: learning rate for the optimizer
    Output
        model, SnakeModel: wrapper exposing fit/predict/save/load
    """
    return SnakeModel(input_size, LR=LR)
