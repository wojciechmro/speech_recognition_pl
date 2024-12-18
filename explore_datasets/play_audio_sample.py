# Run in Interactive Window to display interactive media player
# Make sure you have the Jupyter Notebook extension installed
# Alternatively, play `audio_out/sample.wav` directly

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


if __name__ == "__main__":
    audio_array, sampling_rate, ref_orig = extract_audio_sample_metadata(
        dataset_path="amu-cai/pl-asr-bigos-v2",
        subdir="pwr-azon_spont-20",
        split="train",
        index=0,
    )
    play_audio_sample(audio_array=audio_array, sampling_rate=sampling_rate)
    save_audio_sample(
        audio_array=audio_array,
        sampling_rate=sampling_rate,
        file_path=os.path.join("audio_out", "sample.wav"),
    )
    print(f"Ground truth:\n{ref_orig}")
