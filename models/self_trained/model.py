import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset

from dataset import ASRDataset, collate_fn
from residual import ResidualBlock
from loss import WeightedCTCLoss

class ASRModel(nn.Module):
    def __init__(self, n_mels, num_classes, hidden_dim=512):
        super(ASRModel, self).__init__()

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
            dropout=0.3
        )
        transformer_layer = nn.TransformerEncoderLayer(d_model=hidden_dim * 2, nhead=8, dim_feedforward=1024, dropout=0.3, batch_first=True)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=4)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x):
        x = self.cnn(x) # (batch_size, 128, n_mels//8, time_frames//8)
        x = x.permute(0, 3, 1, 2).reshape(x.size(0), x.size(3), -1) # (batch_size, time_frames, input_size)
        x, _ = self.rnn(x) # (batch_size, time_steps, hidden_dim*2)
        x = self.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.fc(x)
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
criterion = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

def greedy_decoder(preds):
    decoded = []
    for pred in preds:
        decoded_seq = []
        prev_token = None
        for token in pred:
            if token != prev_token and token != 0:
                decoded_seq.append(token)
            prev_token = token
        decoded.append(''.join([inv_vocab[t.item()] for t in decoded_seq]))
    return decoded

# -- Training loop --
def train():
    num_epochs = 3
    is_printed = False
    for epoch in range(num_epochs):
        for batch in train_loader:
            audio = batch["mel_specs"]
            audio_lengths = batch["mel_lengths"]
            adjusted_audio_lengths = audio_lengths // model.downsampling_factor
            transcripts = batch["tokens"]
            transcript_lengths = batch["token_lengths"]
            # Forward pass
            outputs = model(audio)
            log_probs = torch.log_softmax(outputs, dim=-1)
            if not is_printed:
                with torch.no_grad():
                    print("First transcript:", greedy_decoder([transcripts[0]]))
                    preds = torch.argmax(log_probs, dim=-1)
                    print("First pred:", greedy_decoder([preds[0]]))
                    is_printed = True
            # Compute loss
            loss = criterion(
                log_probs.permute(1, 0, 2), # CTC expects (time, batch, num_classes)
                transcripts,
                adjusted_audio_lengths,
                transcript_lengths 
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         print(f"{name}: Gradient Mean = {param.grad.abs().mean()}")
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
        is_printed = False

    torch.save(model.state_dict(), "model_state_dict.pth")

if __name__ == "__main__":
    train()