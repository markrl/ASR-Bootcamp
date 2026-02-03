# ASR Bootcamp
A repo for implementing popular ASR algorithms

## Results
Whisper large v3 turbo (baseline): 16.22

WavLM CTC loss: 13.12
Command:
```python run_wavlm.py --n_workers 16 --finetune_epoch 37 --run_name train_batches0.1 --val_wer --smoothing 0.1 --layer_weights --augment --augment_epoch 45 --logit_dropout_p 0.1 --finetune_lr_mult 1 --min_occurrences 15 --limit_train_batches 0.1```
