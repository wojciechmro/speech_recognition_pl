import warnings

warnings.filterwarnings("ignore", "Trainer.tokenizer is now deprecated.*")

from datasets import load_dataset, DatasetDict

common_voice = DatasetDict()

common_voice["train"] = load_dataset(
    path="amu-cai/pl-asr-bigos-v2",
    name="mozilla-common_voice_15-23",
    split="train",
)
common_voice["test"] = load_dataset(
    path="amu-cai/pl-asr-bigos-v2",
    name="mozilla-common_voice_15-23",
    split="validation",
)
common_voice["train"] = common_voice["train"].shuffle(seed=42).select(range(10000))
common_voice["test"] = common_voice["test"].shuffle(seed=42).select(range(500))

print(common_voice)

common_voice = common_voice.select_columns(["audio", "ref_orig"])

from transformers import WhisperProcessor

processor = WhisperProcessor.from_pretrained(
    "openai/whisper-tiny",
    language="pl",
    task="transcribe",
)
processor.tokenizer.pad_token = processor.tokenizer.eos_token

# common_voice["train"].features

from datasets import Audio

sampling_rate = processor.feature_extractor.sampling_rate
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=sampling_rate))

# example = common_voice["train"][0]

# print(example)

# len_of_audio = len(example["audio"]["array"]) / example["audio"]["sampling_rate"]

# print(len_of_audio)


def prepare_dataset(example):
    audio = example["audio"]
    example = processor(
        audio=audio["array"],
        sampling_rate=audio["sampling_rate"],
        text=example["ref_orig"],
    )
    example["input_length"] = len(audio["array"]) / audio["sampling_rate"]
    return example


common_voice = common_voice.map(
    prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=10
)


def is_audio_in_length_range(length):
    max_input_length = 30.0
    return length < max_input_length


common_voice["train"] = common_voice["train"].filter(
    is_audio_in_length_range,
    input_columns=["input_length"],
)

import torch

device = (
    torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
)
print(f"Using device: {device}")

from dataclasses import dataclass
from typing import Any, Dict, List, Union


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [
            {"input_features": feature["input_features"][0]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

import evaluate

metric = evaluate.load("wer")

from transformers.models.whisper.english_normalizer import BasicTextNormalizer

normalizer = BasicTextNormalizer()


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    print(f"\nProcessing batch of {len(pred_ids)} predictions...")

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # decode predictions and labels
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    # filter out empty references before computing WER
    valid_pred_str = []
    valid_label_str = []
    for p, l in zip(pred_str, label_str):
        if len(l.strip()) > 0:  # only keep non-empty references
            valid_pred_str.append(p)
            valid_label_str.append(l)

    print(f"Filtered out {len(pred_str) - len(valid_pred_str)} empty references")

    # compute orthographic wer
    wer_ortho = 100 * metric.compute(
        predictions=valid_pred_str, references=valid_label_str
    )

    # Print a sample for monitoring
    if len(valid_pred_str) > 0:
        print(f"Sample prediction: {valid_pred_str[0][:100]}...")
        print(f"Sample reference: {valid_label_str[0][:100]}...")
        print(f"Current WER: {wer_ortho:.4f}")

    # compute normalised WER
    pred_str_norm = [normalizer(pred) for pred in valid_pred_str]
    label_str_norm = [normalizer(label) for label in valid_label_str]

    wer = 100 * metric.compute(predictions=pred_str_norm, references=label_str_norm)

    return {"wer_ortho": wer_ortho, "wer": wer}


from transformers import WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny").to(
    device
)

from functools import partial

# disable cache during training since it's incompatible with gradient checkpointing
model.config.use_cache = False

# set language and task for generation and re-enable cache
model.generate = partial(
    model.generate,
    # language="pl",
    # task="transcribe",
    use_cache=True,
)

from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-tiny-pl",  # name on the HF Hub
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=500,
    max_steps=4000,  # increase to 4000 if you have your own GPU or a Colab paid plan
    gradient_checkpointing=True,
    fp16=False,
    evaluation_strategy="steps",
    per_device_eval_batch_size=4,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=500,
    eval_steps=500,
    logging_strategy="steps",
    logging_first_step=True,
    logging_steps=25,
    logging_dir="./logs",
    log_level="warning",
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)
from transformers import Seq2SeqTrainer, EarlyStoppingCallback

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

trainer.train()


from datasets import Dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Reload the test dataset to restore missing columns
test_dataset = load_dataset(
    path="amu-cai/pl-asr-bigos-v2",
    name="mozilla-common_voice_15-23",
    split="validation",
)

# Shuffle and select the subset to match training
test_dataset = test_dataset.shuffle(seed=42).select(range(500))

# Load the fine-tuned model and processor
finetuned_model = WhisperForConditionalGeneration.from_pretrained(
    "whisper-tiny-pl/checkpoint-100"
).to(device)
finetuned_processor = WhisperProcessor.from_pretrained("whisper-tiny-pl/checkpoint-100")

# Load the original model and processor
original_model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-tiny"
).to(device)
original_processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")


# Function to evaluate a model
def evaluate_model(model, processor, dataset, metric):
    predictions, references = [], []
    for example in dataset:
        # Preprocess inputs
        inputs = processor(
            audio=example["audio"]["array"],
            sampling_rate=example["audio"]["sampling_rate"],
            return_tensors="pt",
        ).to(device)
        # Generate predictions
        predicted_ids = model.generate(inputs["input_features"])
        prediction = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        predictions.append(prediction)
        references.append(example["ref_orig"])

    # Normalize and compute WER
    normalized_predictions = [normalizer(p) for p in predictions]
    normalized_references = [normalizer(r) for r in references]
    return metric.compute(
        predictions=normalized_predictions, references=normalized_references
    )


# Filter out examples with empty references
test_dataset = test_dataset.filter(lambda x: x["ref_orig"].strip() != "")

# Check if there are valid examples remaining
if len(test_dataset) == 0:
    print("No valid examples in the test dataset after filtering. Evaluation skipped.")
else:
    # Evaluate fine-tuned model
    fine_tuned_wer = evaluate_model(
        finetuned_model, finetuned_processor, test_dataset, metric
    )
    print(f"Fine-tuned model WER: {fine_tuned_wer:.2f}%")

    # Evaluate original model
    original_wer = evaluate_model(
        original_model, original_processor, test_dataset, metric
    )
    print(f"Original model WER: {original_wer:.2f}%")

    # Compare results
    if fine_tuned_wer < original_wer:
        print("The fine-tuned model performs better than the original Whisper model.")
    else:
        print("The original Whisper model performs better than the fine-tuned model.")
