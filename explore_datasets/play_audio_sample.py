from datasets import load_dataset
from IPython.display import Audio, display


def play_sample_audio(dataset_path, subdir, split, index):
    dataset = load_dataset(path=dataset_path, name=subdir)
    split = dataset[split]
    record = split[index]
    audio_array = record["audio"]["array"]
    sampling_rate = record["audio"]["sampling_rate"]
    return audio_array, sampling_rate


if __name__ == "__main__":
    audio_array, sampling_rate = play_sample_audio(
        dataset_path="amu-cai/pl-asr-bigos-v2",
        subdir="pwr-azon_spont-20",
        split="train",
        index=1,
    )
    display(Audio(data=audio_array, rate=sampling_rate))
