import torch
import json
from torch.utils.data import DataLoader
from datasets import load_dataset

from train import ASRModel, ASRDataset, collate_fn, greedy_decoder

# -- Validation dataset

val_ds = load_dataset(
    path="amu-cai/pl-asr-bigos-v2",
    name="mozilla-common_voice_15-23",
    split="validation"
)

with open("vocab.json", "r") as f:
    vocab_dict = json.load(f)
print("Vocabulary loaded from vocab.json")

n_mels = 80

validation_dataset = ASRDataset(
    dataset=val_ds, 
    sample_rate=16000, 
    n_mels=n_mels,
    max_audio_length=4,
    min_audio_length=0,
    vocab_dict=vocab_dict
)
validation_loader = DataLoader(
    validation_dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=collate_fn
)

num_classes = len(vocab_dict) + 1
model = ASRModel(n_mels, num_classes)

model.load_state_dict(torch.load("model_state_dict.pth", weights_only=True))

# -- Evaluation --

def evaluate(model, val_loader, criterion):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            audio = batch["mel_specs"]
            audio_lengths = batch["mel_lengths"] // 4
            transcripts = batch["tokens"]
            transcript_lengths = batch["token_lengths"]

            # Forward pass
            outputs = model(audio)
            log_probs = torch.log_softmax(outputs, dim=-1)

            print("First transcript:", greedy_decoder([transcripts[0]]))
            preds = torch.argmax(log_probs, dim=-1)
            print("First pred:", greedy_decoder([preds[0]]))

            loss = criterion(
                log_probs.permute(1, 0, 2),
                transcripts,
                audio_lengths,
                transcript_lengths
            )
            total_loss += loss.item() * audio.size(0)
            total_samples += audio.size(0)

        avg_loss = total_loss / total_samples
        return avg_loss


if __name__ == "__main__":
    criterion = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    validation_loss = evaluate(model, validation_loader, criterion)
    print(f"Validation loss: {validation_loss:4f}")
