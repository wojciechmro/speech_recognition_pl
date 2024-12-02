NOTE:

Assign yourself to a task by adding your name within the square brackets of a task. Mark done task with `x`.

TODO:

- [wojtek] explore the BIGOS dataset
  - [ ] assess quality of the recordings
    - [ ] come up with data processing pipeline (cleaning, normalization, feature extraction etc.)
- [ ] streamline repository setup after cloning
  - [ ] create a script that will install train, dev and test sets (make sure they are place in .gitignore)
- [ ] fine-tune existing asr model on one of the subsets (subdirectories) of the training set
  - [ ] find asr model on hugging face that is multilingual (supports Polish) and is relativaly small
  - [ ] assess the performance of this model using WER and CER metrics, evaluate on one of the subsets (subdirectories) of the dev set
    - [ ] make sure the model works well (but not the best) for transcribing Polish speech, we need to have some space for improvement
  - [ ] fine-tune it on one of the subsets (subdirectories) of the training set
  - [ ] evaluate it on the same subset (subdirectory) of the dev set as before
    - [ ] check if the performance is better than before fine-tuning
- [ ] repeat fine tuning on more data (maybe whole training set)
- [ ] train asr model from scratch
