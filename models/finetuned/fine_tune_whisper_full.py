import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from torch.optim import AdamW
import random
from tqdm import tqdm

class WhisperCustomDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.mel_files = [f for f in os.listdir(data_dir) if f.endswith('_log_mel_spectrogram.npy')]
        self.mel_files.sort(key=lambda x: int(x.split('_')[1]))
        self.transcript_files = [f"transcription_{i}.txt" for i in range(len(self.mel_files))]
        
    def __len__(self):
        return len(self.mel_files)
        
    def __getitem__(self, idx):
        mel_path = os.path.join(self.data_dir, self.mel_files[idx])
        mel = np.load(mel_path)
        transcript_path = os.path.join(self.data_dir, self.transcript_files[idx])
        with open(transcript_path, 'r') as f:
            transcript = f.read().strip()
        mel_tensor = torch.from_numpy(mel).float()
        return {
            'input_features': mel_tensor,
            'labels': transcript
        }

def collate_fn(batch):
    input_features = [item['input_features'] for item in batch]
    labels = [item['labels'] for item in batch]
    processed_features = []
    for feat in input_features:
        if feat.shape[1] > 3000:
            processed_feat = feat[:, :3000]
        elif feat.shape[1] < 3000:
            padding = torch.zeros((feat.shape[0], 3000 - feat.shape[1]))
            processed_feat = torch.cat([feat, padding], dim=1)
        else:
            processed_feat = feat
        processed_features.append(processed_feat)
    input_features = torch.stack(processed_features)
    return {
        'input_features': input_features,
        'labels': labels
    }

def save_finetuned_model(model, processor, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Model and processor saved to {output_dir}")

def main():
    random_seed = 42
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    train_dir = "datasets_processed/train"
    full_dataset = WhisperCustomDataset(train_dir)
    total_samples = len(full_dataset)
    subset_size = int(total_samples)
    subset_indices = random.sample(range(total_samples), subset_size)
    train_dataset = Subset(full_dataset, subset_indices)
    batch_size = 8
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="polish", task="transcribe")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=1e-5)
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch in progress_bar:
            input_features = batch['input_features'].to(device)
            labels = processor(text=batch['labels'], return_tensors="pt", padding=True).input_ids
            labels = labels.to(device)
            outputs = model(input_features=input_features, labels=labels)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1} - Average Loss: {avg_loss:.4f}')
    output_dir = "models/finetuned/whisper_finetuned_tiny_full"
    save_finetuned_model(model, processor, output_dir)

if __name__ == "__main__":
    main()