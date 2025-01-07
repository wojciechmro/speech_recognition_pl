import torch.nn as nn


class ASRModel(nn.Module):
    def __init__(self, n_mels, num_classes, hidden_dim=256):
        """
        Initialize an LSTM model for ASR.

        Args:
            n_mels (int): Number of mel frequency bins.
            num_classes (int): Number of output classes (tokens).
            hidden_dim (int): Hidden dimension size for LSTM.
        """
        super(ASRModel, self).__init__()
        # Extract features from mel spectrograms using CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        # Downsample mel spectrograms by a factor of 4
        self.downsampling_factor = 4
        reduced_n_mels = n_mels // self.downsampling_factor
        input_size = 64 * reduced_n_mels
        # Process the sequence of CNN-encoded features with LSTM
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        # Map LSTM output to token probabilities
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # Extract features from mel spectrograms using CNN
        x = self.cnn(x)  # (batch_size, 64, n_mels//4, time_frames//4)
        x = x.permute(0, 3, 1, 2)  # (batch_size, time_frames//4, 64, n_mels//4)
        # Reshape features for LSTM input
        batch_size, time_steps, num_channels, reduced_n_mels = x.size()
        input_size = num_channels * reduced_n_mels
        x = x.reshape(batch_size, time_steps, input_size)
        # Process the sequence of features with LSTM
        x, _ = self.rnn(x)  # (batch_size, time_steps, hidden_dim*2)
        x = self.fc(x)  # (batch_size, time_steps, num_classes)
        return x
