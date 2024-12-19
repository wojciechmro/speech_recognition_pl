import os
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
import torch


processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
    language="polish", task="transcribe"
)
print(f"Special tokens:\n{processor.batch_decode(model.config.forced_decoder_ids)}")

ds = load_dataset(
    path="amu-cai/pl-asr-bigos-v2",
    name="pwr-azon_spont-20",
    split="train",
)
sample = ds[0]["audio"]
input_features = processor(
    sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt"
).input_features


def plot_mel_spectrogram(input_features: torch.Tensor) -> None:
    """Visualize input audio representation for Whisper."""
    import matplotlib.pyplot as plt

    mel_spectrogram = input_features[0].numpy()
    plt.figure(figsize=(12, 6))
    plt.imshow(mel_spectrogram, aspect="auto", origin="lower", cmap="viridis")
    plt.colorbar(label="Amplitude")
    plt.xlabel("Time Steps")
    plt.ylabel("Mel Frequency Bands")
    plt.title("Mel-Spectrogram")
    plt.show()


plot_mel_spectrogram(input_features)

predicted_ids = model.generate(input_features)
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)


def clean_transcription(transcription: str) -> str:
    """Lowercase, remove punctuation and trim whitespace."""
    import string

    return (
        transcription.lower()
        .translate(str.maketrans("", "", string.punctuation))
        .strip()
    )


cleaned_transcription = clean_transcription(transcription[0])
print(f"Cleaned transcription:\n{cleaned_transcription}")

# Inference on 1 sample
os.makedirs("text_out", exist_ok=True)
with open(os.path.join("text_out", "transcription.txt"), "w") as f:
    f.write(cleaned_transcription)

# Inference on 10 samples
for i in range(10):
    sample = ds[i]["audio"]
    input_features = processor(
        sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt"
    ).input_features
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    cleaned_transcription = clean_transcription(transcription[0])
    os.makedirs(os.path.join("text_out", "batch"), exist_ok=True)
    with open(os.path.join("text_out", "batch", f"transcription_{i}.txt"), "w") as f:
        f.write(cleaned_transcription)
