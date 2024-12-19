import os
from datasets import load_dataset
from IPython.display import Audio, display
import soundfile as sf


def extract_audio_sample_metadata(dataset_path, subdir, split, index):
    dataset = load_dataset(path=dataset_path, name=subdir, split=split)
    record = dataset[index]
    ref_orig = record["ref_orig"]
    audio_array = record["audio"]["array"]
    sampling_rate = record["audio"]["sampling_rate"]
    return audio_array, sampling_rate, ref_orig


def play_audio_sample(audio_array, sampling_rate):
    display(Audio(data=audio_array, rate=sampling_rate))


def save_audio_sample(audio_array, sampling_rate, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    sf.write(file=file_path, data=audio_array, samplerate=sampling_rate)


def save_reference_text(text, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        f.write(text)


if __name__ == "__main__":
    # Metadata extraction on 1 sample
    audio_array, sampling_rate, ref_orig = extract_audio_sample_metadata(
        dataset_path="amu-cai/pl-asr-bigos-v2",
        subdir="pwr-azon_spont-20",
        split="train",
        index=0,
    )
    print(f"Ground truth for only 1 sample:\n{ref_orig}")
    print("---")
    save_audio_sample(
        audio_array=audio_array,
        sampling_rate=sampling_rate,
        file_path=os.path.join("audio_out", "sample.wav"),
    )
    save_reference_text(
        text=ref_orig, file_path=os.path.join("text_out", "transcription.txt")
    )

    # Metadata extraction on 10 samples
    for i in range(10):
        audio_array, sampling_rate, ref_orig = extract_audio_sample_metadata(
            dataset_path="amu-cai/pl-asr-bigos-v2",
            subdir="pwr-azon_spont-20",
            split="train",
            index=i,
        )
        print(f"Ground truth for sample {i}:\n{ref_orig}")
        save_audio_sample(
            audio_array=audio_array,
            sampling_rate=sampling_rate,
            file_path=os.path.join("audio_out", "batch", f"sample_{i}.wav"),
        )
        save_reference_text(
            text=ref_orig,
            file_path=os.path.join("text_out", "batch", f"transcription_{i}.txt"),
        )
