from enum import Enum
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import numpy as np
from evaluation_metrics import calculate_CER, calculate_WER
import os


class WhisperModel(Enum):
    STANDARD = "standard"
    SUBSET_FINETUNED = "subset"
    FULL_FINETUNED = "full"


def get_model_config(model_type: WhisperModel):
    """
    Get model configuration based on the selected model type

    Args:
        model_type (WhisperModel): Enum value indicating which model to use

    Returns:
        tuple: (model_path, output_file_name, model_description)
    """
    configs = {
        WhisperModel.STANDARD: (
            "openai/whisper-tiny",
            "whisper_tiny_no_finetuning.txt",
            "Standard Model",
        ),
        WhisperModel.SUBSET_FINETUNED: (
            "models/finetuned/whisper_finetuned_tiny",
            "whisper_tiny_finetuned_on_subset.txt",
            "Subset Fine-tuned Model",
        ),
        WhisperModel.FULL_FINETUNED: (
            "models/finetuned/whisper_finetuned_tiny_full",
            "whisper_tiny_finetuned_on_full_dataset.txt",
            "Full Dataset Fine-tuned Model",
        ),
    }
    return configs[model_type]


def transcribe_polish_audio(audio_path, processor, model):
    """
    Transcribe Polish audio using Whisper tiny model (standard or fine-tuned)

    Args:
        audio_path (str): Path to the audio file
        processor: WhisperProcessor instance
        model: WhisperForConditionalGeneration instance

    Returns:
        str: Transcribed text
    """
    # Load and preprocess audio
    audio, sr = librosa.load(audio_path, sr=16000)

    # Convert to float32 and normalize
    audio = audio.astype(np.float32)
    if np.abs(audio).max() > 1.0:
        audio = audio / np.abs(audio).max()

    # Process audio
    input_features = processor(
        audio, sampling_rate=16000, return_tensors="pt"
    ).input_features

    # Force Polish language
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="pl", task="transcribe"
    )

    # Generate transcription
    predicted_ids = model.generate(
        input_features, forced_decoder_ids=forced_decoder_ids, language="pl"
    )

    # Decode the transcription
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    return transcription


def evaluate_whisper_transcriptions(model_type: WhisperModel):
    """
    Evaluate Whisper transcriptions on all WAV files in the validation dataset

    Args:
        model_type (WhisperModel): Enum value indicating which model to use
    """
    # Setup paths
    validation_dir = os.path.join("ETL", "datasets", "validation")
    output_dir = os.path.join("models_eval", "logs_of_results")

    # Get model configuration
    model_path, output_filename, model_description = get_model_config(model_type)
    output_file = os.path.join(output_dir, output_filename)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load model and processor
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)

    # Lists to store metrics
    all_cer = []
    all_wer = []

    # Prepare output file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Evaluation Results ({model_description})\n")
        f.write("=" * (len(model_description) + 21) + "\n\n")

        # Process each WAV file
        all_wav_files = [
            f
            for f in os.listdir(validation_dir)
            if f.startswith("sample_") and f.endswith(".wav")
        ]
        wav_files = sorted([f for f in all_wav_files if "sample_4.wav" not in f])

        for wav_file in wav_files:
            # Get corresponding transcript file
            file_number = wav_file.split("_")[1].split(".")[0]
            transcript_file = os.path.join(
                validation_dir, f"transcription_{file_number}.txt"
            )

            # Skip if transcript doesn't exist
            if not os.path.exists(transcript_file):
                print(f"Warning: No transcript found for {wav_file}")
                continue

            # Get reference text
            with open(transcript_file, "r", encoding="utf-8") as tf:
                reference = tf.read().strip()

            # Get prediction
            prediction = transcribe_polish_audio(
                os.path.join(validation_dir, wav_file), processor, model
            )

            # Calculate metrics
            cer = calculate_CER(reference, prediction)
            wer = calculate_WER(reference, prediction)

            # Store metrics
            all_cer.append(cer)
            all_wer.append(wer)

            # Write detailed results
            f.write("--------------------------------------------------\n")
            f.write(f"File: {wav_file}\n")
            f.write(f"Reference: {reference}\n")
            f.write(f"Prediction: {prediction}\n")
            f.write(f"CER: {cer:.4f}\n")
            f.write(f"WER: {wer:.4f}\n")
            f.write("--------------------------------------------------\n\n")

        # Calculate and write averages
        avg_cer = np.mean(all_cer)
        avg_wer = np.mean(all_wer)

        # Write summary at the beginning of the file
        with open(output_file, "r", encoding="utf-8") as f:
            content = f.read()

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"Evaluation Results ({model_description})\n")
            f.write("=" * (len(model_description) + 21) + "\n")
            f.write(f"Average CER: {avg_cer:.4f}\n")
            f.write(f"Average WER: {avg_wer:.4f}\n\n")
            f.write("Detailed Results:\n")
            f.write("================\n")
            f.write(
                content[
                    content.find("--------------------------------------------------") :
                ]
            )


if __name__ == "__main__":
    # NOTE: execute the following files before running this script:
    # - `ETL/load_data.py`
    # - `ETL/preprocess_data.py`
    # - `models/finetuned/fine_tune_whisper.py`
    os.chdir(os.path.dirname(os.path.dirname(__file__)))  # go one level up
    evaluate_whisper_transcriptions(WhisperModel.STANDARD)
    evaluate_whisper_transcriptions(WhisperModel.SUBSET_FINETUNED)
    evaluate_whisper_transcriptions(WhisperModel.FULL_FINETUNED)
