# Spoken Digit Recognition

A deep learning system that listens to short audio clips and predicts which digit (0–9) was spoken. Built entirely from scratch — no pretrained models used.

---

## Problem Statement

Given a short `.wav` audio file of a person saying a digit, predict which digit it is. The challenge: the test set contains **unseen speakers** and **noisy/augmented recordings**, so the model must learn what a digit *sounds like*, not who is saying it.

- 10-class classification (digits 0–9)
- ~54,000 audio clips total
- Evaluated on generalization across speakers and noise conditions

---

## Results

| Metric | Value |
|---|---|
| Validation Accuracy | **99.3%** |
| Public Leaderboard Score | **0.993** |
| Training Time | ~12 minutes |
| Test Predictions | 16,200 clips |

---

## Tech Stack

- **Language:** Python
- **Framework:** PyTorch, torchaudio
- **Features:** Log-Mel Spectrogram (64 mel bands, 128 time frames)
- **Augmentation:** SpecAugment (frequency + time masking)
- **Optimizer:** Adam (lr=1e-3, weight_decay=1e-4)
- **Scheduler:** Cosine Annealing LR
- **Hardware:** NVIDIA T4 GPU (Kaggle)

---

## Model Architecture

Custom CNN trained from scratch — input is treated as a (1, 64, 128) grayscale image.

```
Input: Log-Mel Spectrogram (1 × 64 × 128)
│
├── Conv Block 1: Conv2D(32) → BN → ReLU → Conv2D(32) → BN → ReLU → MaxPool → Dropout(0.2)
├── Conv Block 2: Conv2D(64) → BN → ReLU → Conv2D(64) → BN → ReLU → MaxPool → Dropout(0.2)
├── Conv Block 3: Conv2D(128) → BN → ReLU → MaxPool → Dropout(0.3)
│
├── AdaptiveAvgPool2d(4×4)
├── Flatten → Linear(2048 → 256) → ReLU → Dropout(0.5)
└── Linear(256 → 10) → Output
```

---

## Project Structure

```
├── train_audio/          # Training .wav files
├── test_audio/           # Test .wav files
├── train.csv             # Training labels (id, label)
├── notebook.ipynb        # Full training notebook (Kaggle)
└── submission.csv        # Final predictions
```

---

## Pipeline

### 1. Feature Extraction
Raw audio → resampled to 16kHz → mono → **Log-Mel Spectrogram**
- 64 mel bands, FFT size 512, hop length 160
- Normalized per sample (mean=0, std=1)
- Padded/cropped to fixed 128 time frames

### 2. Caching
All spectrograms precomputed and saved as `.pt` files before training — eliminates redundant computation across 30 epochs.

### 3. Augmentation (Training Only)
**SpecAugment** applied on spectrograms:
- Frequency masking: up to 10 mel bins zeroed out
- Time masking: up to 20 time frames zeroed out

Simulates the noisy and distorted conditions in the test set.

### 4. Training
- 30 epochs, batch size 128
- Adam optimizer with cosine annealing LR scheduler
- Best model saved by validation accuracy
- 90/10 train/val split

---

## Training Log (Key Epochs)

| Epoch | Train Acc | Val Acc |
|---|---|---|
| 1 | 79.0% | 95.3% |
| 5 | 96.9% | 98.8% |
| 15 | 98.6% | 99.2% |
| 27 | 99.5% | 99.3% |
| 30 | 99.6% | 99.3% |

---

## Rules Compliance

This project was built under strict competition constraints:

- ✅ Trained from scratch — no pretrained weights (no wav2vec2, HuBERT, Whisper, etc.)
- ✅ Only provided training data used — no external datasets
- ✅ No manual labeling of test samples
- ✅ No internet downloads during training
- ✅ Fully reproducible Kaggle notebook

---

## Key Design Decisions

| Decision | Reasoning |
|---|---|
| Log-Mel over raw waveform | Matches human auditory perception, enables CNN on 2D image |
| 64 mel bands | Enough frequency detail without excess computation |
| SpecAugment | Directly simulates noisy/partial test conditions |
| Spectrogram caching | 30–40% speedup per epoch by avoiding recomputation |
| Cosine LR schedule | Smooth decay allows fine-grained convergence in later epochs |
| Dropout 0.2/0.5 | Prevents memorizing speaker identity instead of digit patterns |
