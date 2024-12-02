NOTE:

Assign yourself to a task by adding your name within the square brackets of a task. Mark done task with `x`.

TODO:

- [ ] fine-tune existing asr model
  - [ ] find asr model on hugging face that is multilingual (supports Polish) and is relativaly small
  - [ ] assess the performance of the model using WER and CER metrics, evaluate on one of the subsets (subdirectories) of the dev set
    - [ ] make sure the model works well (but not the best) for transcribing Polish speech, we need to have some space for improvement
  - [ ] fine-tune it on one of the subsets (subdirectories) of the training set
  - [ ] evaluate it on the same subset (subdirectory) of the dev set as before
    - [ ] check if the performance is better than before fine-tuning
