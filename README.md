# CS230 Final Project: Audio-Based Head Pose Estimation

This repository contains implementations of deep learning models for predicting head poses from microphone array audio. The project explores multiple neural network architectures to estimate both positional (x, y, z) and rotational (quaternion: qx, qy, qz, qw) head tracking using the EasyCom dataset.

## Team Members

- **Jaduk Suh** - Sequential Models (LSTM & Transformer)
- **Prerana Rane** - Baseline Models (MLP) & CRNN Models
- **Sebastian Preprelita** - CNN Models (1D & 2D)

## Dataset

This project uses the [EasyCom Dataset](https://imperialcollegelondon.github.io/spear-challenge/data), which contains:
- **Audio**: 6-channel microphone array audio recorded at 48 kHz
- **Poses**: Head tracking data sampled at 20 Hz (position + quaternion orientation)
- **Sessions**: 12 sessions split into train (sessions 1-10), validation (session 11), and test (session 12)
- **Annotations**: Speech transcriptions for filtering active speech frames

## Project Structure

```
src/
├── baseline/          # MLP baseline model (Prerana Rane)
├── crnn/              # CRNN models (Prerana Rane)
├── CNN/               # CNN models (Sebastian Preprelita)
├── sequential/        # Sequential models (Jaduk Suh)
└── utils/             # Shared utilities
```