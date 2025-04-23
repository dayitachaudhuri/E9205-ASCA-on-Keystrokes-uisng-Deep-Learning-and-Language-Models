from glob import glob
import json
import os
import traceback
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf

def extract_label(filepath, full_dataset=False):
    if full_dataset:
        label = os.path.basename(os.path.dirname(filepath))
    else:
        label = os.path.splitext(os.path.basename(filepath))[0]
    return label

def load_wav_files(dataset_path):
    audio_data = []
    labels = [] 
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                try:
                    signal, sr = librosa.load(file_path, sr=None)
                    audio_data.append(signal)
                    label = extract_label(file_path)
                    labels.append(label)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    return audio_data, labels

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

def time_shift(segment, max_shift_percentage=0.4):
    # Calculate the maximum shift in samples
    max_shift = int(len(segment) * max_shift_percentage)
    # Take a random integer shift within the range [-max_shift, max_shift]
    shift = np.random.randint(-max_shift, max_shift)
    # Shift the segment
    shifted_segment = np.roll(segment, shift)
    # Zero out the wrapped portion to avoid artifacts
    if shift > 0:
        shifted_segment[:shift] = 0
    elif shift < 0:
        shifted_segment[shift:] = 0
    return shifted_segment

def generate_mel_spectrogram(segment, sr, n_mels=64, n_fft=1024, hop_length=225):
    S = librosa.feature.melspectrogram(y=segment, sr=sr, n_fft=n_fft,  hop_length=hop_length, n_mels=n_mels)
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB

def spec_augment(spectrogram, time_mask_percentage=0.1, freq_mask_percentage=0.1):
    augmented_spec = spectrogram.copy()
    n_mels, n_time = augmented_spec.shape

    # Calculate mask lengths
    time_mask_length = max(1, int(n_time * time_mask_percentage))
    freq_mask_length = max(1, int(n_mels * freq_mask_percentage))

    # Mask Value (mean of spectrogram)
    mask_value = np.mean(augmented_spec)

    # Apply Time Mask: Select a random start point and mask time steps across all frequencies
    time_start = np.random.randint(0, n_time - time_mask_length + 1)
    augmented_spec[:, time_start:time_start + time_mask_length] = mask_value

    # Apply Frequency Mask: Select a random start point and mask frequency bins across all time steps
    freq_start = np.random.randint(0, n_mels - freq_mask_length + 1)
    augmented_spec[freq_start:freq_start + freq_mask_length, :] = mask_value

    return augmented_spec

def process_audio_file(file_path, energy_threshold, output_dir, segment_length=14400, sr=44100):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Load the audio file
    audio, sr = librosa.load(file_path, sr=sr)
    # Isolate keystrokes
    segments = isolate_keystrokes(audio, sr, threshold=energy_threshold, segment_length=segment_length)
    for idx, segment in enumerate(segments, start=1):  # start from 1
        # Apply time shift for data augmentation
        augmented_segment = time_shift(segment)
        # Save audio as 1.wav, 2.wav, ...
        segment_path = os.path.join(output_dir, f"{idx}.wav")
        sf.write(segment_path, augmented_segment, sr)
    return len(segments)


# Load config
with open('config.json', 'r') as f:
    config = json.load(f)
input_root = config['DATASET_PATH']['all']      
output_root = config['OUTPUT_PATHS']['all']    
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
                energy_threshold=100,
                output_dir=output_dir,
                segment_length=14400,
                sr=44100
            )

            print(f"[SUCCESS] Processed {device_file} into {segment_count} segments")

        except Exception as e:
            print(f"[ERROR] Failed to process {device_file}")
            traceback.print_exc()