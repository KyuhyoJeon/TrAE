import torch
import torch.nn as nn

class TrLSTM(nn.Module):
    def __init__(self, hidden_size=512, num_layers=5):
        super(TrLSTM, self).__init__()
        self.embed = nn.Linear(1, hidden_size)
        self.layer = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.embed_inv = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.embed(x.unsqueeze(-1))
        x = self.embed(x)
        x, _ = self.layer(x)
        self.embed_inv.weight = nn.Parameter(torch.linalg.pinv(self.embed.weight))
        x = self.embed_inv(x).squeeze(-1)
        
        return x