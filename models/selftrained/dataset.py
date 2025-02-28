import torch
import torchaudio.transforms as T
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import unicodedata


def get_inv_vocab(vocab_dict):
    """Invert a vocabulary dictionary to map indices back to characters. Used to decode predicted tokens back to text."""
    return {idx: char for char, idx in vocab_dict.items()}


class ASRDataset(Dataset):
    def __init__(
        self,
        dataset,
        sample_rate=16000,
        n_mels=80,
        n_fft=400,
        hop_length=160,
        max_audio_length=None,
        min_audio_length=0,
        vocab_dict=None,
    ):
        """
        Initialize PyTorch Dataset class for audio-to-text tasks.

        Args:
            dataset (list): List of dictionaries, each containing audio and transcript data.
            sample_rate (int, optional): Sample rate for audio processing. Defaults to 16000.
            n_mels (int, optional): Number of mel frequency bins. Defaults to 80.
            n_fft (int, optional): Number of FFT bins. Defaults to 400.
            hop_length (int, optional): Hop length for STFT. Defaults to 160.
            max_audio_length (int, optional): Maximum audio length in seconds. Defaults to None.
            min_audio_length (int, optional): Minimum audio length in seconds. Defaults to 0.
            vocab_dict (dict, optional): Dictionary mapping characters to indices. Defaults to None.
        """
        # Initialize mel spectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
        )
        self.sample_rate = sample_rate
        # Filter dataset based on audio length
        if max_audio_length:
            max_samples = int(max_audio_length * sample_rate)
            min_samples = int(min_audio_length * sample_rate)
            self.dataset = [
                item
                for item in dataset
                if len(item["audio"]["array"]) <= max_samples
                and len(item["audio"]["array"]) >= min_samples
            ]
            print(
                f"Filtered dataset to {len(self.dataset)} from {len(dataset)} samples."
            )
        else:
            self.dataset = dataset
        # Create vocabulary if not provided
        if vocab_dict:
            self.vocab_dict = vocab_dict
            self.inv_vocab = get_inv_vocab(vocab_dict)
        else:
            self.vocab_dict, self.inv_vocab = self._create_vocab()

    def normalize_transcript(self, transcript):
        """Normalize transcript. Remove punctuation, convert to lowercase."""
        transcript = transcript.lower()  # lowercase
        transcript = "".join(
            char
            for char in transcript
            if not unicodedata.category(char).startswith("P")  # remove punctuation
        )
        return transcript

    def _create_vocab(self):
        """Create vocabulary and inverse vocabulary from transcripts."""
        transcripts = [
            self.normalize_transcript(item["ref_orig"]) for item in self.dataset
        ]
        all_chars = "".join(transcripts)
        unique_chars = sorted(set(all_chars))
        vocab_dict = {char: idx + 1 for idx, char in enumerate(unique_chars)}
        vocab_dict["<blank>"] = 0  # add blank token for CTC loss
        inv_vocab = get_inv_vocab(vocab_dict)
        print(f"Generated vocabulary with {len(vocab_dict)} tokens.")
        return vocab_dict, inv_vocab

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Get item from dataset.

        Args:
            idx (int): Index of the item to get.

        Returns:
            dict: Dictionary containing mel spectrogram, mel length, tokens, and token length.
        """
        # Convert audio to tensor and normalize to [-1, 1]
        audio = torch.tensor(self.dataset[idx]["audio"]["array"], dtype=torch.float32)
        audio = audio / audio.abs().max()
        # Get transcript and normalize
        transcript = self.dataset[idx]["ref_orig"]
        transcript = self.normalize_transcript(transcript)
        # Compute mel spectrogram and add batch dimension using unsqueeze
        mel_spec = self.mel_transform(audio).unsqueeze(0)
        # Turn transcript into token indices
        tokens = []
        for char in transcript:
            if char in self.vocab_dict:
                tokens.append(self.vocab_dict[char])
            else:
                print(f"Warning: Character '{char}' not in vocab_dict.")
        return {
            "mel_spec": mel_spec,
            "mel_length": mel_spec.shape[2],  # time axis
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "token_length": len(tokens),
        }


def collate_fn(batch):
    """Pad mel spectrograms and tokens to the same length."""
    # Get max time axis length
    max_time = max(item["mel_spec"].shape[2] for item in batch)
    # Pad mel spectrograms to the same length
    mel_specs = [
        F.pad(item["mel_spec"], (0, max_time - item["mel_spec"].shape[2]))
        for item in batch
    ]
    # Pad tokens to the same length
    tokens = [item["tokens"] for item in batch]
    # Get mel lengths and token lengths
    mel_lengths = [item["mel_length"] for item in batch]
    token_lengths = [item["token_length"] for item in batch]
    # Stack mel spectrograms and tokens
    mel_specs_padded = torch.stack(mel_specs, dim=0)
    tokens_padded = pad_sequence(tokens, batch_first=True)
    return {
        "mel_specs": mel_specs_padded,
        "mel_lengths": torch.tensor(mel_lengths, dtype=torch.long),
        "tokens": tokens_padded,
        "token_lengths": torch.tensor(token_lengths, dtype=torch.long),
    }


def greedy_decoder(preds, inv_vocab):
    """Convert model predictions (token indices) into readable text. Remove blank tokens and join characters."""
    decoded = []
    for pred in preds:
        decoded_seq = []
        prev_token = None
        for token in pred:
            if token != prev_token and token != 0:
                decoded_seq.append(token)
            prev_token = token
        decoded.append("".join([inv_vocab[t.item()] for t in decoded_seq]))
    return decoded
