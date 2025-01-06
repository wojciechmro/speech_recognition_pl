# Goal

Develop a Polish ASR (Automatic Speech Recognition) system capable of transcribing spoken words into text.

### Motivation

Solve the 3rd challange from the PolEval 2024 competition.

### Task

Find task definition, datasets and evaluation metrics [here](https://beta.poleval.pl/challenge/2024-asr-bigos).

### Deadline

11.01.2025

# Notes

### Division Of Labor

[Roadmap of tasks](https://github.com/users/wojciechmro/projects/2/views/4)

- each task is marked for one day for simplicity
- only order of tasks matters
- tasks that can be done asynchronously have deadline for the same day
- assign yourself to tasks
- append `(#<issue_number>)` to the commit message to link it to the task (see example commits using `git log`)

### Pretrained ASR Models

**Whisper**

- capable of transcription and translation
  - supports 96 languages
- trained on 680,000 hours of labeled speech data
  - audio inputs were 30 seconds long and resampled to 16kHz
  - then converted to 80-channel log-magnitude Mel spectrogram representation (computed on 25ms windows with a stride of 10ms)
  - then all values in this feature space (log-Mel spectrogram) are scaled to [-1, 1] range
 
## Audio Preprocessing

Ensure n_fft and hop_length are set correctly relative to the sample rate. For a sample rate of 16kHz:
  - n_fft=400 corresponds to a window of 25ms.
  - hop_length=160 corresponds to a stride of 10ms.
