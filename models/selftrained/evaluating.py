import torch
import json
from torch.utils.data import DataLoader
from datasets import load_dataset
from lstm import ASRModel
from dataset import ASRDataset, collate_fn, greedy_decoder, get_inv_vocab

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from models_eval.eval_metrics import calculate_CER, calculate_WER

# Load vocabulary
with open("vocab.json", "r") as f:
    vocab_dict = json.load(f)
inv_vocab = get_inv_vocab(vocab_dict)
print("Vocabulary loaded from vocab.json")
# Create data loader for validation dataset
val_ds = load_dataset(
    path="amu-cai/pl-asr-bigos-v2",
    name="mozilla-common_voice_15-23",
    split="validation",
)
n_mels = 80
validation_dataset = ASRDataset(
    dataset=val_ds,
    sample_rate=16000,
    n_mels=n_mels,
    max_audio_length=6,
    min_audio_length=0,
    vocab_dict=vocab_dict,
)
validation_loader = DataLoader(
    validation_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn
)
# Initialize model and load weights
num_classes = len(vocab_dict) + 1
model = ASRModel(n_mels, num_classes)
model.load_state_dict(torch.load("model_state_dict.pth", weights_only=True))


# Evaluate model
def evaluate(model, val_loader, criterion):
    """Evaluate ASR model on the validation dataset.

    Args:
        model (ASRModel): ASR model to evaluate.
        val_loader (DataLoader): Data loader for the validation dataset.
        criterion (torch.nn.Module): Loss function to use for evaluation.

    Returns:
        float: Average loss on the validation dataset.
    """
    model.eval()  # set model to evaluation mode
    total_loss = 0.0
    total_samples = 0
    all_cer = 0.0
    all_wer = 0.0
    with torch.no_grad():  # disable gradient computation
        for batch in val_loader:
            audio = batch["mel_specs"]
            audio_lengths = batch["mel_lengths"] // 4
            transcripts = batch["tokens"]
            transcript_lengths = batch["token_lengths"]
            # Forward pass
            outputs = model(audio)
            log_probs = torch.log_softmax(outputs, dim=-1)
            preds = torch.argmax(log_probs, dim=-1)
            # Compute loss
            loss = criterion(
                log_probs.permute(1, 0, 2),
                transcripts,
                audio_lengths,
                transcript_lengths,
            )
            total_loss += loss.item() * audio.size(0)
            total_samples += audio.size(0)

            references = greedy_decoder(transcripts, inv_vocab)
            predictions = greedy_decoder(preds, inv_vocab)
            for i, reference in enumerate(references):
                all_cer += calculate_CER(reference, predictions[i])
                all_wer += calculate_WER(reference, predictions[i])

        avg_loss = total_loss / total_samples
        avg_cer = all_cer / total_samples
        avg_wer = all_wer / total_samples
        return avg_loss, avg_cer, avg_wer


if __name__ == "__main__":
    criterion = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    validation_loss, cer, wer = evaluate(model, validation_loader, criterion)
    print(f"Validation loss: {validation_loss:4f}")
    print(f"CER: {cer:.4f}\n")
    print(f"WER: {wer:.4f}\n")
