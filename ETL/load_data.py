import os
from datasets import load_dataset
import soundfile as sf

DATASET_DIR = "datasets"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
DEV_DIR = os.path.join(DATASET_DIR, "validation")
TEST_DIR = os.path.join(DATASET_DIR, "test")

BIGOS_DATASET = "amu-cai/pl-asr-bigos-v2"
CONFIG = "pwr-azon_spont-20"

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

def download_and_process_datasets():
    print(f"Downloading train data for {BIGOS_DATASET} with config {CONFIG}...")
    bigos_train = load_dataset(BIGOS_DATASET, CONFIG, split="train")
    
    for i, record in enumerate(bigos_train):
        audio_array = record["audio"]["array"]
        sampling_rate = record["audio"]["sampling_rate"]
        ref_orig = record["ref_orig"]
        
        audio_file_path = os.path.join(TRAIN_DIR, f"sample_{i}.wav")
        text_file_path = os.path.join(TRAIN_DIR, f"transcription_{i}.txt")
        
        save_audio_sample(audio_array, sampling_rate, audio_file_path)
        save_reference_text(ref_orig, text_file_path)

    print(f"Downloading dev data for {BIGOS_DATASET} with config {CONFIG}...")
    bigos_dev = load_dataset(BIGOS_DATASET, CONFIG, split="validation")
    
    for i, record in enumerate(bigos_dev):
        audio_array = record["audio"]["array"]
        sampling_rate = record["audio"]["sampling_rate"]
        ref_orig = record["ref_orig"]
        
        audio_file_path = os.path.join(DEV_DIR, f"sample_{i}.wav")
        text_file_path = os.path.join(DEV_DIR, f"transcription_{i}.txt")
        
        save_audio_sample(audio_array, sampling_rate, audio_file_path)
        save_reference_text(ref_orig, text_file_path)

    print(f"Downloading test data for {BIGOS_DATASET} with config {CONFIG}...")
    bigos_test = load_dataset(BIGOS_DATASET, CONFIG, split="test")
    
    for i, record in enumerate(bigos_test):
        audio_array = record["audio"]["array"]
        sampling_rate = record["audio"]["sampling_rate"]
        ref_orig = record["ref_orig"]
        
        audio_file_path = os.path.join(TEST_DIR, f"sample_{i}.wav")
        text_file_path = os.path.join(TEST_DIR, f"transcription_{i}.txt")
        
        save_audio_sample(audio_array, sampling_rate, audio_file_path)
        save_reference_text(ref_orig, text_file_path)

def main():
    create_directories()
    download_and_process_datasets()
    print("Setup complete!")

if __name__ == "__main__":
    main()
