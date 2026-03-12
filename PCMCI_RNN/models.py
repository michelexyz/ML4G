
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


class SimpleRNN(nn.Module):
    """
    RNN or LSTM for regression, returning a single value per sample.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 16, rnn_type='RNN'):
        super().__init__()
        if rnn_type == 'RNN':
            self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        else:
            raise ValueError("rnn_type must be 'RNN' or 'LSTM'")
        
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        """
        x: shape (batch_size, seq_len, input_dim)
        """
        out, _ = self.rnn(x)
        # Take the last hidden state
        last_hidden = out[:, -1, :]  # (batch_size, hidden_dim)
        pred = self.fc(last_hidden)  # (batch_size, 1)
        return pred.squeeze(-1)


def train_model(
    model: nn.Module, 
    X_train: torch.Tensor, 
    Y_train: torch.Tensor,
    epochs: int = 20, 
    lr: float = 1e-3,
    batch_size: int = 32
):
    """
    Fits the model to (X_train, Y_train) using mini-batches.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Build a DataLoader for mini-batched training
    dataset = TensorDataset(X_train, Y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        model.train()
        for Xb, Yb in dataloader:
            optimizer.zero_grad()
            preds = model(Xb)
            loss = loss_fn(preds, Yb)
            loss.backward()
            optimizer.step()


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray, train_mean: float) -> float:
    """
    Coefficient of determination
    """
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - train_mean)**2)

    if ss_res > ss_tot:
        print("Warning: negative R^2")
        return 0.0
    return 1.0 - ss_res/ss_tot if ss_tot > 1e-12 else 0.0