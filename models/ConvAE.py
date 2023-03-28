import torch.nn as nn

class ConvEncoder(nn.Module):
    def __init__(self, input_channel, channels, output_channel):
        super(ConvEncoder, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(input_channel, channels[0], 3, 2, 1, bias=False),
            nn.BatchNorm1d(channels[0], affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels[0], channels[1], 3, 2, 1, bias=False),
            nn.BatchNorm1d(channels[1], affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels[1], channels[2], 3, 2, 1, bias=False),
            nn.BatchNorm1d(channels[2], affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels[2], channels[3], 3, 2, 1, bias=False),
            nn.BatchNorm1d(channels[3], affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels[3], output_channel, 3, 2, 1, bias=True),
        )

    def forward(self, x):
        x = self.model(x)
        return x
    

class ConvDecoder(nn.Module):
    def __init__(self, input_channel, channels, output_channel):
        super(ConvDecoder, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose1d(output_channel, channels[3], 3, 2, 1, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(channels[3], channels[2], 3, 2, 1, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(channels[2], channels[1], 3, 2, 1, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(channels[1], channels[0], 3, 2, 1, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(channels[0], input_channel, 3, 2, 1, 1, bias=True),
        )

    def forward(self, x):
        x = self.model(x)
        return x

class Model(nn.Module):
    def __init__(self, input_channel, channels, output_channel):
        super(Model, self).__init__()
        self.encoder = ConvEncoder(input_channel, channels, output_channel)
        self.decoder = ConvDecoder(input_channel, channels, output_channel)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        
        return x