import Levenshtein


def calculate_CER(reference_text: str, predicted_text: str) -> float:
    """Calculate Character Error Rate (CER)."""
    num_errors = Levenshtein.distance(reference_text, predicted_text)
    cer = num_errors / len(reference_text) if len(reference_text) > 0 else 0.0
    return round(cer, 4)


def calculate_WER(reference_text: str, predicted_text: str) -> float:
    """Calculate Word Error Rate (WER)."""
    reference_words = reference_text.split()
    predicted_words = predicted_text.split()
    num_errors = Levenshtein.distance(reference_words, predicted_words)
    wer = num_errors / len(reference_words) if len(reference_words) > 0 else 0.0
    return round(wer, 4)
