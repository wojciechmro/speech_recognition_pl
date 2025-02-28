import os
import shutil
import librosa
import numpy as np
import soundfile as sf
import librosa.display

ORIGINAL_DATASET_DIR = "datasets"
PROCESSED_DATASET_DIR = "datasets_processed"
TRAIN_DIR = os.path.join(PROCESSED_DATASET_DIR, "train")
DEV_DIR = os.path.join(PROCESSED_DATASET_DIR, "validation")
TEST_DIR = os.path.join(PROCESSED_DATASET_DIR, "test")
TARGET_SAMPLE_RATE = 16000
MFCC_FEATURES = 13


def create_processed_directories():
    """Creates necessary directories for processed data."""
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(DEV_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)
    print("Processed directories created.")


def clean_denoise(audio_array):
    """Apply basic denoising using librosa."""
    return librosa.effects.preemphasis(audio_array)


def normalize_audio(audio_array, sample_rate):
    """Normalize audio to a constant sample rate and scale the volume to -1 to 1 range."""
    audio_resampled = librosa.resample(
        audio_array, orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE
    )
    return librosa.util.normalize(audio_resampled)


def extract_mfcc_features(audio_array, sample_rate):
    """Extract MFCC features from audio."""
    return librosa.feature.mfcc(y=audio_array, sr=sample_rate, n_mfcc=MFCC_FEATURES)


def extract_log_mel_spectrogram(audio_array, sample_rate, n_mels=80, hop_length=160):
    """Extract log-mel spectrogram from audio."""
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio_array, sr=sample_rate, n_mels=n_mels, hop_length=hop_length
    )
    return librosa.power_to_db(mel_spectrogram, ref=np.max)


def save_processed_audio(audio_array, sample_rate, file_path):
    """Save the processed audio to a file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    sf.write(file_path, audio_array, samplerate=sample_rate)


def save_mfcc_features(mfcc, file_path):
    """Save MFCC features to a numpy file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    np.save(file_path, mfcc)


def save_spectrogram(spectrogram, file_path):
    """Save spectrogram to a numpy file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    np.save(file_path, spectrogram)


def copy_transcription_files(input_dir, output_dir):
    """Copy transcription files to the output directory (only for train and validation)."""
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".txt") and file.startswith("transcription"):
                transcription_file_path = os.path.join(root, file)
                output_file_path = transcription_file_path.replace(
                    input_dir, output_dir
                )
                os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                shutil.copy(transcription_file_path, output_file_path)


def process_and_save_files(input_dir, output_dir):
    """Process all audio files in the specified directory."""
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                audio_array, sample_rate = librosa.load(file_path, sr=None)
                audio_cleaned = clean_denoise(audio_array)
                audio_normalized = normalize_audio(audio_cleaned, sample_rate)
                processed_audio_path = file_path.replace(input_dir, output_dir)
                save_processed_audio(
                    audio_normalized, TARGET_SAMPLE_RATE, processed_audio_path
                )
                mfcc = extract_mfcc_features(audio_normalized, TARGET_SAMPLE_RATE)
                mfcc_file_path = processed_audio_path.replace(".wav", "_mfcc.npy")
                save_mfcc_features(mfcc, mfcc_file_path)
                spectrogram = extract_log_mel_spectrogram(
                    audio_normalized, TARGET_SAMPLE_RATE
                )
                spectrogram_file_path = processed_audio_path.replace(
                    ".wav", "_log_mel_spectrogram.npy"
                )
                save_spectrogram(spectrogram, spectrogram_file_path)


def process_data():
    """Main function to process all datasets."""
    create_processed_directories()
    print("Processing train data...")
    process_and_save_files(os.path.join(ORIGINAL_DATASET_DIR, "train"), TRAIN_DIR)
    copy_transcription_files(os.path.join(ORIGINAL_DATASET_DIR, "train"), TRAIN_DIR)
    print("Processing validation data...")
    process_and_save_files(os.path.join(ORIGINAL_DATASET_DIR, "validation"), DEV_DIR)
    copy_transcription_files(os.path.join(ORIGINAL_DATASET_DIR, "validation"), DEV_DIR)
    print("Processing test data...")
    process_and_save_files(os.path.join(ORIGINAL_DATASET_DIR, "test"), TEST_DIR)
    print("Data processing complete!")


if __name__ == "__main__":
    process_data()
