import os
from evaluation_metrics import calculate_CER, calculate_WER

reference_path = os.path.join("..", "explore_datasets", "text_out", "transcription.txt")
predicted_path = os.path.join(
    "..", "models", "pretrained", "text_out", "transcription.txt"
)

with open(predicted_path, "r") as f:
    predicted_text = f.read()
with open(reference_path, "r") as f:
    reference_text = f.read()

cer = calculate_CER(reference_text, predicted_text)
wer = calculate_WER(reference_text, predicted_text)
print(f"Character Error Rate (CER):\n{cer}")
print(f"Word Error Rate (WER):\n{wer}")
