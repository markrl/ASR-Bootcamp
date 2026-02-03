# ASR Bootcamp
A repo for implementing popular ASR algorithms

## Results
|           Model                   | WER(%) |
|-----------------------------------|--------|
| Whisper large v3 turbo (baseline) | 16.22  |
|        WavLM CTC loss             | 11.72  |

WavLM CTC loss Command:

```python run_wavlm.py --n_workers 16 --finetune_epoch 15 --run_name train_batches1.0 --val_wer --smoothing 0.1 --layer_weights --augment --augment_epoch 18 --logit_dropout_p 0.1 --finetune_lr_mult 1 --min_occurrences 15 --limit_train_batches 1.0 --limit_val_batches 1.0```
