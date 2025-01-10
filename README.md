# Project Goals

1. Develop a Polish ASR (Automatic Speech Recognition) system from scratch.
2. Fine-tune existing ASR model (Whisper) to improve its base performance.

### Motivation

Solve the 3rd challange from the PolEval 2024 competition.

### Task

Find task definition, datasets and evaluation metrics [here](https://beta.poleval.pl/challenge/2024-asr-bigos).

### Division Of Labor

[Roadmap of tasks](https://github.com/users/wojciechmro/projects/2/views/4)

# Documentation

### Navigating the Repository

#### Exploring the Dataset

Utility files for exploring the dataset are located in the `explore_datasets/` directory.

#### Evaluation Metrics

Evaluation metrics are implemented in the `models_eval/eval_metrics.py` file.

#### Training ASR model from scratch

Find code for training and evaluating a Polish ASR model from scratch in the `models/selftrained` directory.

#### Fine tuning

Fine tuning was done using two approaches:

1. First approach uses `etl/load_data.py` and `etl/preprocess_data.py` to download and preprocess the dataset. The model is then fine tuned using `models/finetuned/fine_tune_whisper.py` and evaluated with `models_eval/eval_whisper.py`.

2. Second approach is solely implemented in `models/finetuned_v2/main.py`.

#### Reproducing data and model weights

The data and model weights were to large to be uploaded to the repository and are ignored through the `.gitignore` file.

To run the code locally, create a virtual environment and install the dependencies from the `requirements.txt` file like this:

```bash
python -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

### Theoretical Background

#### Whisper (our inspiration)

- capable of transcription and translation
  - supports 96 languages
  - uses special tokens to represent target task and language (e.g. <|transcribe|>, <|translate|>, <|en|>, <|pl|>)
- trained on 680,000 hours of labeled speech data
  - audio inputs were 30 seconds long and resampled to 16kHz
  - then converted to 80-channel log-magnitude Mel spectrogram representation
    - computed on 25ms windows (n_fft=400) with a stride of 10ms (hop_length=160)
  - then all values in this feature space (log-Mel spectrogram) are scaled to [-1, 1] range
- trained with the Seq2Seq architecture where the input sequence (log-mel spectrogram) is encoded into a hidden representation, then decoded step-by-step (autoregressively) to produce tokens
  - uses attention mechanisms to focus on the relevant parts of the encoded sequence at each decoding step
- uses cross-entropy loss to train the model

#### Whisper Fine-tuning

In order to fine-tune the Whisper model, we had to:

- preprocess the dataset with the same techniques (16kHz, >=30seconds, normalize amplitude levels, convert audio to log-mel spectrograms with the same parameters i.e. window size of 25ms and stride of 10ms)
- use the same tokenizer as the original Whisper model
- use the same loss function as the original Whisper model (cross-entropy loss)
- use the same optimizer as the original Whisper model (AdamW)
- use small learning rate as the original Whisper model (1e-5) to avoid drastic changes in the model weights

#### ASR Training From Scratch

- We initially considered training our own transformer using a Seq2Seq architecture, but such models require massive amounts of training data and computational resources that were beyond our scope
- Instead, we developed a hybrid CNN-LSTM model using the CTC (Connectionist Temporal Classification) architecture:
  - Input: Log-mel spectrograms representing frequency bins over time frames
  - CNN Front-End: Convolutional layers downsample the input and extract local acoustic features
  - Bidirectional LSTM: Processes the sequential data to capture both past and future context
  - Dense Layer: Projects to character probabilities (including blank token)
  - CTC Loss: Handles variable-length alignment between audio and text

The hybrid architecture combines CNNs for efficient local pattern extraction (formants, harmonics) with bidirectional LSTMs for temporal context modeling. During inference, we use greedy decoding to convert frame-wise probabilities into text by selecting the most likely tokens and merging repeats/blanks.

#### Evaluation Metrics

We evaluated our models using two standard metrics for ASR:

##### Word Error Rate (WER)

$\text{WER} = \frac{S + D + I}{N}$

Where:

- $S$ = number of substitutions
- $D$ = number of deletions
- $I$ = number of insertions
- $N$ = total number of words in reference

WER measures word-level accuracy by counting the minimum number of word edits needed to transform the hypothesis into the reference text, normalized by the reference length.

##### Character Error Rate (CER)

$\text{CER} = \frac{S_c + D_c + I_c}{N_c}$

Where:

- $S_c$ = number of character substitutions
- $D_c$ = number of character deletions
- $I_c$ = number of character insertions
- $N_c$ = total number of characters in reference

CER is similar to WER but operates at the character level, providing a more granular view of transcription accuracy. It's especially useful for agglutinative languages like Polish where small character errors can significantly impact word accuracy.

# Results
