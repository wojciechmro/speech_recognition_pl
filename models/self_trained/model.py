import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchaudio
import torchaudio.transforms as T
from datasets import load_dataset
from collections import Counter

from dataset import ASRDataset, collate_fn

class ASRModel(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=256):
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
        self.rnn = None
        self.fc = None
    
    def forward(self, x):
        # x: (batch_size, 1, n_mels, time_frames)
        batch_size, _, n_mels, time_frames = x.size()

        x = self.cnn(x) # (batch_size, 64, n_mels//4, time_frames//4)
        x = x.permute(0, 3, 1, 2) # (batch_size, time_frames//4, 64, n_mels//4)

        batch_size, time_steps, num_channels, reduced_n_mels = x.size()
        input_size = num_channels * reduced_n_mels
        x = x.reshape(batch_size, time_steps, input_size)

        if self.rnn is None or self.fc is None:
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=self.hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
            self.fc = nn.Linear(self.hidden_dim * 2, self.num_classes)
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

train_dataset = ASRDataset(ds, vocab_dict)
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=collate_fn
)

# -- Evaluation --

def greedy_decoder(log_probs):
    preds = torch.argmax(log_probs, dim=-1)
    decoded = []
    for pred in preds:
        decoded_seq = []
        prev_token = None
        for token in pred:
            if token != prev_token and token != 0:
                decoded_seq.append(token)
            prev_token = token
        decoded.append(''.join([inv_vocab[t.item()] for t in decoded_seq]))
    print(decoded)
    return decoded

# -- Training loop --
# Model hyperparameters
freq_dim = 128
input_dim = 64 * (freq_dim // 4)
num_classes = len(vocab) + 1
model = ASRModel(input_dim, num_classes)

# Optimizer and loss
optimizer = optim.Adam(model.parameters(), lr=1e-3)
ctc_loss = nn.CTCLoss(blank=0)

num_epochs = 5
for epoch in range(num_epochs):
    for batch in train_loader:
        audio = batch["mel_specs"]
        audio_lengths = batch["mel_lengths"]
        transcripts = batch["tokens"]
        transcript_lengths = batch["token_lengths"]

        # Forward pass
        outputs = model(audio)
        log_probs = torch.log_softmax(outputs, dim=-1)

        # Compute loss
        loss = ctc_loss(
            log_probs.permute(1, 0, 2), # CTC expects (time, batch, num_classes)
            transcripts,
            audio_lengths,
            transcript_lengths 
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")



