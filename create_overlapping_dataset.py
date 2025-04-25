import os
import random
import librosa
import soundfile as sf
import numpy as np
from glob import glob
from tqdm import tqdm
import json

INPUT_DIR = 'data/ProcessedWithImage'
OUTPUT_DIR = 'data/KeystrokeCountDataset'
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_KEYSTROKES = 3
SAMPLE_RATE = 16000
SEGMENT_LENGTH = int(0.9 * SAMPLE_RATE)  # 0.9 seconds

def collect_all_keystroke_files():
    keystroke_files = []
    for label in os.listdir(INPUT_DIR):
        label_dir = os.path.join(INPUT_DIR, label)
        if not os.path.isdir(label_dir): continue
        for sub in os.listdir(label_dir):
            if sub.startswith(label):
                wavs = glob(os.path.join(label_dir, sub, "*.wav"))
                keystroke_files.extend(wavs)
    return keystroke_files

def generate_sample(keystroke_files, count):
    mix = np.zeros(SEGMENT_LENGTH)
    positions = sorted(random.sample(range(0, SEGMENT_LENGTH - 4000), count))  # time offsets
    selected = random.sample(keystroke_files, count)
    for i, file in enumerate(selected):
        y, sr = librosa.load(file, sr=SAMPLE_RATE)
        if len(y) > 4000: y = y[:4000]
        start = positions[i]
        end = start + len(y)
        mix[start:end] += y
    # Normalize to avoid clipping
    if np.max(np.abs(mix)) > 1.0:
        mix = mix / np.max(np.abs(mix))
    return mix

def build_dataset(n_samples_per_class=500):
    keystroke_files = collect_all_keystroke_files()
    metadata = []
    for count in range(MAX_KEYSTROKES + 1):
        for i in tqdm(range(n_samples_per_class), desc=f"Generating {count}-keystroke samples"):
            if count == 0:
                audio = np.zeros(SEGMENT_LENGTH)
            else:
                audio = generate_sample(keystroke_files, count)
            filename = f"{count}_{i}.wav"
            path = os.path.join(OUTPUT_DIR, filename)
            sf.write(path, audio, SAMPLE_RATE)
            metadata.append({'file': filename, 'label': count})
    
    with open(os.path.join(OUTPUT_DIR, 'labels.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

if __name__ == '__main__':
    build_dataset(n_samples_per_class=500)