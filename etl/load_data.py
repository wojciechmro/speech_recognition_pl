import os
from datasets import load_dataset
import soundfile as sf

DATASET_DIR = "datasets"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
DEV_DIR = os.path.join(DATASET_DIR, "validation")
TEST_DIR = os.path.join(DATASET_DIR, "test")
BIGOS_DATASET = "amu-cai/pl-asr-bigos-v2"
CONFIG = "pwr-azon_spont-20"
MAX_DURATION_SECONDS = 30


def create_directories():
    print("Creating dataset directories...")
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(DEV_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)


def save_audio_sample(audio_array, sampling_rate, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    sf.write(file=file_path, data=audio_array, samplerate=sampling_rate)


def save_reference_text(text, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        f.write(text)


def process_split(split_name, output_dir):
    print(f"Downloading {split_name} data for {BIGOS_DATASET} with config {CONFIG}...")
    dataset = load_dataset(BIGOS_DATASET, CONFIG, split=split_name)
    total_files = len(dataset)
    print(f"Total files in {split_name} split before filtering: {total_files}")

    sample_count = 0
    filtered_count = 0

    for i, record in enumerate(dataset):
        audio_array = record["audio"]["array"]
        sampling_rate = record["audio"]["sampling_rate"]
        duration = len(audio_array) / sampling_rate

        # Skip files longer than MAX_DURATION_SECONDS
        if duration > MAX_DURATION_SECONDS:
            filtered_count += 1
            continue

        ref_orig = record["ref_orig"]
        audio_file_path = os.path.join(output_dir, f"sample_{sample_count}.wav")
        text_file_path = os.path.join(output_dir, f"transcription_{sample_count}.txt")

        save_audio_sample(audio_array, sampling_rate, audio_file_path)
        save_reference_text(ref_orig, text_file_path)
        sample_count += 1

    print(f"Files in {split_name} split after filtering: {sample_count}")
    print(f"Filtered out {filtered_count} files longer than {MAX_DURATION_SECONDS}s")
    print(f"Percentage of files kept: {(sample_count/total_files)*100:.1f}%")


def download_and_process_datasets():
    splits = [("train", TRAIN_DIR), ("validation", DEV_DIR), ("test", TEST_DIR)]

    for split_name, output_dir in splits:
        process_split(split_name, output_dir)


def main():
    create_directories()
    download_and_process_datasets()
    print("Setup complete!")


if __name__ == "__main__":
    main()
