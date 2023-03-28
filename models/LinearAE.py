import torch.nn as nn


class mlpEncoder(nn.Module):
    def __init__(self, input_channel, channels, output_channel):
        super(mlpEncoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_channel, channels[0], bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channels[0], channels[1], bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channels[1], channels[2], bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channels[2], channels[3], bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channels[3], output_channel, bias=True),
        )

    def forward(self, x):
        x = self.model(x)
        return x


class mlpDecoder(nn.Module):
    def __init__(self, input_channel, channels, output_channel):
        super(mlpDecoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(output_channel, channels[3], bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channels[3], channels[2], bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channels[2], channels[1], bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channels[1], channels[0], bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channels[0], input_channel, bias=True),
        )

    def forward(self, x):
        x = self.model(x)
        return x
    

class TrAE(nn.Module):
    def __init__(self, input_channel, channels, output_channel):
        super(TrAE, self).__init__()
        self.encoder = mlpEncoder(input_channel, channels, output_channel)
        self.decoder = mlpDecoder(input_channel, channels, output_channel)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        
        return x