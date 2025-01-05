import torch
import json
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm


from dataset import ASRDataset, collate_fn, greedy_decoder
from lstm import ASRModel


# -- Creating dataloader

ds = load_dataset(
    path="amu-cai/pl-asr-bigos-v2",
    name="mozilla-common_voice_15-23",
    split="train"
)

n_mels = 80
train_dataset = ASRDataset(
    dataset=ds,  
    sample_rate=16000, 
    n_mels=n_mels,
    max_audio_length=4,
    min_audio_length=0
    )
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=collate_fn
)

# Model hyperparameters
num_classes = len(train_dataset.vocab_dict) + 1
model = ASRModel(n_mels, num_classes)

# Optimizer and loss
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)


# -- Training loop --
def train():
    num_epochs = 10
    is_printed = False
    for epoch in range(num_epochs):
        for batch in tqdm(train_loader, desc="Training"):
            audio = batch["mel_specs"]
            audio_lengths = batch["mel_lengths"] // 4
            transcripts = batch["tokens"]
            transcript_lengths = batch["token_lengths"]
            # Forward pass
            outputs = model(audio)
            log_probs = torch.log_softmax(outputs, dim=-1)
            if not is_printed:
                with torch.no_grad():
                    print("transcripts:", greedy_decoder(transcripts[0:2], train_dataset.inv_vocab))
                    preds = torch.argmax(log_probs, dim=-1)
                    print("preds:", greedy_decoder(preds[0:2], train_dataset.inv_vocab))
                    is_printed = True
            # Compute loss
            loss = criterion(
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
        is_printed = False

    torch.save(model.state_dict(), "model_state_dict.pth")

if __name__ == "__main__":
    train()
    with open("vocab.json", "w") as f:
        json.dump(train_dataset.vocab_dict, f)
    print("Vocabulary saved to vocab.json")