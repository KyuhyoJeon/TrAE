import math
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvEncoder(nn.Module):
    def __init__(self, input_channel, hidden, output_channel):
        super(ConvEncoder, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(input_channel, hidden*(2**0), 3, 2, 1, bias=False),
            nn.BatchNorm1d(hidden*(2**0), affine=True),                             # affine = False
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden*(2**0), hidden*(2**1), 3, 2, 1, bias=False),
            nn.BatchNorm1d(hidden*(2**1), affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden*(2**1), hidden*(2**2), 3, 2, 1, bias=False),
            nn.BatchNorm1d(hidden*(2**2), affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden*(2**2), hidden*(2**1), 3, 2, 1, bias=False),
            nn.BatchNorm1d(hidden*(2**1), affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden*(2**1), output_channel, 3, 2, 1, bias=False),
        )

    def forward(self, x):
        x = self.model(x)
        return x
    

class ConvDecoder(nn.Module):
    def __init__(self, input_channel, hidden, output_channel):
        super(ConvDecoder, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose1d(output_channel, hidden*(2**1), 3, 2, 1, 1, bias=False),  # bias = False
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(hidden*(2**1), hidden*(2**2), 3, 2, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(hidden*(2**2), hidden*(2**1), 3, 2, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(hidden*(2**1), hidden*(2**0), 3, 2, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(hidden*(2**0), input_channel, 3, 2, 1, 1, bias=False),
        )

    def forward(self, x):
        x = self.model(x)
        return x
    
    
class mlpEncoder(nn.Module):
    def __init__(self, input_channel, hidden, output_channel):
        super(mlpEncoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_channel, hidden*(2**0), bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden*(2**0), hidden*(2**1), bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden*(2**1), hidden*(2**2), bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden*(2**2), hidden*(2**1), bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden*(2**1), output_channel, bias=True),
        )

    def forward(self, x):
        x = self.model(x)
        return x


class mlpDecoder(nn.Module):
    def __init__(self, input_channel, hidden, output_channel):
        super(mlpDecoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(output_channel, hidden*(2**1), bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden*(2**1), hidden*(2**2), bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden*(2**2), hidden*(2**1), bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden*(2**1), hidden*(2**0), bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden*(2**0), input_channel, bias=True),
        )

    def forward(self, x):
        x = self.model(x)
        return x
    

class TrAE(nn.Module):
    def __init__(self, encoder, decoder, args):
        super(TrAE, self).__init__()
        self.encoder = encoder(args.input_channel, args.t, args.output_channel)
        self.decoder = decoder(args.input_channel, args.t, args.output_channel)
        
    def forward(self, x):
        hidden = self.encoder(x)
        out = self.decoder(hidden)
        
        return out, hidden


class TrVAE(nn.Module):
    def __init__(self, encoder, decoder, args):
        super(TrAE, self).__init__()
        self.encoder = encoder(args.input_channel, args.t, args.t)
        self.decoder = decoder(args.input_channel, args.t, args.output_channel)
        
        self.fc_mu = nn.Linear(args.t, args.output_channel)
        self.fc_var = nn.Linear(args.t, args.output_channel)
        
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std
        
    def forward(self, x):
        hidden = self.encoder(x)
        mu = self.fc_mu(hidden)
        log_var = self.fc_var(hidden)
        z = self.reparameterize(mu, log_var)
        out = self.decoder(z)
        
        return out, mu, log_var
    

BCE = nn.BCELoss(reduction='sum')
def VAE_loss(x, recon_x, mu, log_var):
    BCE_loss = BCE(recon_x, x.view(-1, 784))
    KLD_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE_loss + KLD_loss 
    

class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Classifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(128, output_dim, bias=False),
        )

    def forward(self, x):
        x = self.model(x)
        return x
    
# class Classifier(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(Classifier, self).__init__()
#         self.model = nn.Linear(input_dim, output_dim)

#     def forward(self, x):
#         x = self.model(x)
#         return x


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

    
class TrLinear(nn.Module):
    def __init__(self, configs):
        super(TrLinear, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.channels = configs.enc_in
        self.individual = configs.individual
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len,self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        if self.individual:
            output = torch.zeros([x.size(0),self.pred_len,x.size(2)],dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:,:,i] = self.Linear[i](x[:,:,i])
            x = output
        else:
            x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
        return x # [Batch, Output length, Channel]