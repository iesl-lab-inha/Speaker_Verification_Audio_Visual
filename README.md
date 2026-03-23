# Audio-Visual Speaker Verification Model

This project implements an audio-visual speaker verification architecture based on:

- Audio branch: FBANK feature extraction + TDNN-BiLSTM encoder
- Visual branch: cropped lip ROI frames + ResNet-18 encoder
- Fusion: gated fusion module
- Temporal modeling: shared Conformer block
- Pooling: multi-head attention pooling
- Embedding: fully connected layer + batch normalization + L2 normalization
- Scoring: cosine similarity for speaker verification

## File Structure

- `utils.py`
- `audio_encoder.py`
- `visual_encoder.py`
- `fusion.py`
- `conformer_block.py`
- `pooling.py`
- `av_model.py`

## Requirements

- Python 3.9+
- PyTorch
- torchvision

```bash
pip install torch torchvision
