import torch
import torchaudio.transforms as T
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import string

class ASRDataset(Dataset):
    def __init__(self, dataset, vocab_dict, sample_rate=16000, n_mels=80, max_audio_length=None):
        self.vocab_dict = vocab_dict
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=400,
            hop_length=160
        )
        self.sample_rate = sample_rate

        if max_audio_length:
            max_samples = int(max_audio_length * sample_rate)
            self.dataset = [
                item for item in dataset
                if len(item["audio"]["array"]) <= max_samples
            ]
            print(f"Filtered dataset to {len(self.dataset)} from {len(dataset)} samples.")
        else:
            self.dataset = dataset
    
    def preprocess_transcript(self, transcript):
        transcript = transcript.lower()
        transcript = transcript.translate(str.maketrans('', '', string.punctuation))
        return transcript

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        audio = torch.tensor(self.dataset[idx]["audio"]["array"], dtype=torch.float32)
        audio = audio / audio.abs().max()
        transcript = self.dataset[idx]["ref_orig"]
        transcript = self.preprocess_transcript(transcript)
        mel_spec = self.mel_transform(audio).unsqueeze(0)
        tokens = []
        for char in transcript:
            if char in self.vocab_dict:
                tokens.append(self.vocab_dict[char])
            else:
                print(f"Warning: Character '{char}' not in vocab_dict.")
        return {
            "mel_spec": mel_spec,
            "mel_length": mel_spec.shape[2],
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "token_length": len(tokens),
        }

def collate_fn(batch):
    max_time = max(item["mel_spec"].shape[2] for item in batch)
    mel_specs = [
        F.pad(item["mel_spec"], (0, max_time - item["mel_spec"].shape[2])) for item in batch]
    tokens = [item["tokens"] for item in batch]
    mel_lengths = [item["mel_length"] for item in batch]
    token_lengths = [item["token_length"] for item in batch]
    mel_specs_padded = torch.stack(mel_specs, dim=0)
    tokens_padded = pad_sequence(tokens, batch_first=True)
    return {
        "mel_specs": mel_specs_padded,
        "mel_lengths": torch.tensor(mel_lengths, dtype=torch.long),
        "tokens": tokens_padded,
        "token_lengths": torch.tensor(token_lengths, dtype=torch.long),
    }
