import os
from eval_metrics import calculate_CER, calculate_WER

cer_list = []
wer_list = []

for i in range(10):
    reference_path = os.path.join(
        "..", "explore_datasets", "text_out", "batch", f"transcription_{i}.txt"
    )
    predicted_path = os.path.join(
        "..", "models", "pretrained", "text_out", "batch", f"transcription_{i}.txt"
    )

    with open(predicted_path, "r") as f:
        predicted_text = f.read()
    with open(reference_path, "r") as f:
        reference_text = f.read()

    cer = calculate_CER(reference_text, predicted_text)
    wer = calculate_WER(reference_text, predicted_text)
    cer_list.append(cer)
    wer_list.append(wer)

print(f"Average Character Error Rate (CER):\n{sum(cer_list) / len(cer_list)}")
print(f"Average Word Error Rate (WER):\n{sum(wer_list) / len(wer_list)}")
