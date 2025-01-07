import torch.nn as nn
from residual import ResidualBlock


class TransformerASR(nn.Module):
    def __init__(self, n_mels, num_classes, hidden_dim=512):
        super(TransformerASR, self).__init__()

        # Feature extractor (CNN with Residual Blocks)
        self.cnn = nn.Sequential(
            ResidualBlock(1, 32, stride=2),
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 128, stride=2),
        )
        self.downsampling_factor = 8
        reduced_n_mels = n_mels // self.downsampling_factor
        input_size = 128 * reduced_n_mels
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim * 2,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.3,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=4)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = self.cnn(x)  # (batch_size, 128, n_mels//8, time_frames//8)
        x = x.permute(0, 3, 1, 2).reshape(
            x.size(0), x.size(3), -1
        )  # (batch_size, time_frames, input_size)
        x, _ = self.rnn(x)  # (batch_size, time_steps, hidden_dim*2)
        x = self.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.fc(x)
        return x
