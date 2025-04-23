import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import torch
import librosa
import json
from CoatNet import CoAtNet
from AudioKeystrokeDataset import isolate_keystrokes, generate_mel_spectrogram

# === CONFIGURATION ===
WAV_PATH     = "data/sentences/1.wav"
MODEL_PATH   = "models/best_model.pth"
LABEL_PATH   = "data/label2idx.json"

SR               = 44100
N_FFT            = 1024
HOP_LENGTH_STFT  = 256
SEGMENT_LENGTH   = 14400
N_MELS           = 64
SPEC_HOP_LENGTH  = 500
THRESHOLD        = 100

# === 1) Load label2idx and build idx2label ===
with open(LABEL_PATH, 'r') as f:
    label2idx = json.load(f)
idx2label = {idx: label for label, idx in label2idx.items()}
num_classes = len(label2idx)
print(f"Loaded {num_classes} unique class labels from '{LABEL_PATH}'")

# === 2) Load model and weights ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CoAtNet(num_classes=num_classes).to(device)
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# === 3) Inference on continuous sentence ===
audio, _ = librosa.load(WAV_PATH, sr=SR)
segments = isolate_keystrokes(
    audio,
    sr=SR,
    threshold=THRESHOLD,
    segment_length=SEGMENT_LENGTH,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH_STFT
)
if not segments:
    raise RuntimeError("No keystroke segments detected. Try adjusting THRESHOLD.")

pred_chars = []
with torch.no_grad():
    for seg in segments:
        mel = generate_mel_spectrogram(seg, sr=SR)
        mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-6)
        x = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).to(device)
        out = model(x)
        idx = out.argmax(dim=1).item()
        pred_chars.append(idx2label[idx])

sentence = "".join(pred_chars)
print("Predicted sentence:", sentence)
