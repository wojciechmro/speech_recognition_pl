import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset

from dataset import ASRDataset, collate_fn

class ASRModel(nn.Module):
    def __init__(self, n_mels, num_classes, hidden_dim=256):
        super(ASRModel, self).__init__()

        # Feature extractor (CNN)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        reduced_n_mels = n_mels // 4
        input_size = 64 * reduced_n_mels
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x):
        # x: (batch_size, 1, n_mels, time_frames)
        batch_size, _, n_mels, time_frames = x.size()

        x = self.cnn(x) # (batch_size, 64, n_mels//4, time_frames//4)
        x = x.permute(0, 3, 1, 2) # (batch_size, time_frames//4, 64, n_mels//4)

        batch_size, time_steps, num_channels, reduced_n_mels = x.size()
        input_size = num_channels * reduced_n_mels
        x = x.reshape(batch_size, time_steps, input_size)

        x, _ = self.rnn(x) # (batch_size, time_steps, hidden_dim*2)
        x = self.fc(x) # (batch_size, time_steps, num_classes)
        return x

# -- Creating vocab

ds = load_dataset(
    path="amu-cai/pl-asr-bigos-v2",
    name="pwr-azon_spont-20",
    split="train"
)

transcripts = [example["ref_orig"] for example in ds]

all_charts = ''.join(transcripts)
vocab = sorted(set(all_charts))
vocab_dict = {char: idx + 1 for idx, char in enumerate(vocab)}
vocab_dict['<blank>'] = 0
inv_vocab = {idx: char for char, idx in vocab_dict.items()}

# -- Creating dataloader

n_mels = 80
train_dataset = ASRDataset(ds, vocab_dict, sample_rate=16000, n_mels=n_mels)
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=collate_fn
)

# Model hyperparameters
num_classes = len(vocab) + 1
model = ASRModel(n_mels, num_classes)

# Optimizer and loss
optimizer = optim.Adam(model.parameters(), lr=1e-3)
ctc_loss = nn.CTCLoss(blank=0)

# -- Training loop --
def train():
    num_epochs = 1
    for epoch in range(num_epochs):
        for batch in train_loader:
            audio = batch["mel_specs"]
            audio_lengths = batch["mel_lengths"]
            downsampling_factor = 4
            adjusted_audio_lengths = audio_lengths // downsampling_factor
            transcripts = batch["tokens"]
            transcript_lengths = batch["token_lengths"]

            # Forward pass
            outputs = model(audio)
            log_probs = torch.log_softmax(outputs, dim=-1)

            # Compute loss
            loss = ctc_loss(
                log_probs.permute(1, 0, 2), # CTC expects (time, batch, num_classes)
                transcripts,
                adjusted_audio_lengths,
                transcript_lengths 
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    torch.save(model.state_dict(), "model_state_dict.pth")

if __name__ == "__main__":
    train()