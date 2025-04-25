from glob import glob
import json
import os
import traceback
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
from tqdm import tqdm 
from torch.utils.data import Dataset

def extract_label_and_device(filepath, full_dataset=False):
    filepath = filepath.replace("\\", "/")  # Normalize path
    if full_dataset:
        # Example: data/All Dataset/Raw Data/<label>/<label><device>.wav
        label = os.path.basename(os.path.dirname(filepath)).lower()
        filename = os.path.splitext(os.path.basename(filepath))[0].lower()
        if filename.startswith(label):
            device = filename[len(label):]  # Remove only the prefix
        else:
            device = "unknown"
    else:
        # Example: data/<device>/Raw Data/<label>.wav
        parts = filepath.split("/")
        device = parts[-3].lower()
        label = os.path.splitext(os.path.basename(filepath))[0].lower()
    return label, device

def isolate_keystrokes(audio, sr, segment_length=14400, n_fft=1024, hop_length=256, min_separation=43):
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    energy = np.sum(np.abs(stft), axis=0)
    threshold = np.percentile(energy, 95) 
    min_separation = int(0.1 * sr / hop_length) 
    keystroke_frames = np.where(energy > threshold)[0]
    if len(keystroke_frames) == 0:
        return threshold, [], []
    groups = []
    current_group = [keystroke_frames[0]]
    for i in range(1, len(keystroke_frames)):
        if keystroke_frames[i] - keystroke_frames[i - 1] > min_separation:
            groups.append(current_group)
            current_group = []
        current_group.append(keystroke_frames[i])
    if current_group:
        groups.append(current_group)
    segments = []
    segment_starts = []
    for group in groups:
        center_frame = int(np.mean(group))
        start = max((center_frame * hop_length) - segment_length // 2, 0)
        end = start + segment_length
        if end > len(audio):
            start = max(0, len(audio) - segment_length)
            end = len(audio)
        segment = audio[start:end]
        if len(segment) == segment_length: 
            segments.append(segment)
            segment_starts.append(start / sr) 
    return threshold, segments, segment_starts

def visualize_keystrokes(audio, sr, segment_starts, title="Detected Keystrokes"):
    plt.figure(figsize=(12, 3))
    librosa.display.waveshow(audio, sr=sr)
    for start_time in segment_starts:
        plt.axvline(start_time, color='red', linestyle='--')
    plt.title(title)
    plt.show()

def generate_mel_spectrogram(segment, sr, n_mels=64, n_fft=1024, hop_length=225):
    S = librosa.feature.melspectrogram(y=segment, sr=sr, n_fft=n_fft,  hop_length=hop_length, n_mels=n_mels)
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB

def process_audio_file(file_path, output_dir, segment_length=14400, sr=44100):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Load the audio file
    audio, sr = librosa.load(file_path, sr=sr)
    # Isolate keystrokes
    threshold, segments, segment_starts = isolate_keystrokes(audio, sr, segment_length=segment_length)
    # Visualize and save the waveform with keystroke markers
    label, _ = extract_label_and_device(file_path)
    plt.figure(figsize=(12, 3))
    visualize_keystrokes(audio, sr, segment_starts, title=f"{label} Keystrokes")
    vis_path = os.path.join(output_dir, f"{label}.png")
    plt.savefig(vis_path)
    plt.close()
    # Save segments as individual wavs
    for idx, segment in enumerate(segments, start=1):
        segment_path = os.path.join(output_dir, f"{idx}.wav")
        sf.write(segment_path, segment, sr)
    return len(segments)

with open('config.json', 'r') as f:
    config = json.load(f)
input_root = config['DATASET_PATH']['all']      
output_root = 'data/ProcessedWithImage'  
print(f"[INFO] Loaded paths from config.")

# Traverse key directories
key_dirs = [d for d in glob(os.path.join(input_root, '*')) if os.path.isdir(d)]
print(f"[INFO] Found {len(key_dirs)} keys: {[os.path.basename(k) for k in key_dirs]}")

for key_dir in key_dirs:
    key_name = os.path.basename(key_dir)
    device_files = glob(os.path.join(key_dir, '*.wav')) + glob(os.path.join(key_dir, '*.WAV'))

    if not device_files:
        print(f"[WARNING] No .wav files in {key_dir}")
        continue

    for device_file in device_files:
        try:
            device_name = os.path.splitext(os.path.basename(device_file))[0]
            output_dir = os.path.join(output_root, key_name, device_name)
            os.makedirs(output_dir, exist_ok=True)

            print(f"[INFO] Processing: {device_file}")
            print(f"[INFO] Output dir: {output_dir}")

            # Process file
            segment_count = process_audio_file(
                file_path=device_file,
                output_dir=output_dir,
                segment_length=14400,
                sr=44100
            )

            print(f"[SUCCESS] Processed {device_file} into {segment_count} segments")

        except Exception as e:
            print(f"[ERROR] Failed to process {device_file}")
            traceback.print_exc()