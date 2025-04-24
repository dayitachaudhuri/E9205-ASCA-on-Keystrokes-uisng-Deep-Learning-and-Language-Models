import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
from tqdm import tqdm 
from torch.utils.data import Dataset
import torch

def extract_label_and_device(filepath, full_dataset=False):
    filepath = filepath.replace("\\", "/")
    parts = filepath.split("/")

    if "ProcessedWithImage" in parts:
        label = parts[-3].lower()
        combined = parts[-2].lower()
        if combined.startswith(label):
            device = combined[len(label):]
        else:
            device = "unknown"
        # Debug print
    elif full_dataset:
        label = os.path.basename(os.path.dirname(filepath)).lower()
        filename = os.path.splitext(os.path.basename(filepath))[0].lower()
        if filename.startswith(label):
            device = filename[len(label):] 
        else:
            device = "unknown"
    else:
        parts = filepath.split("/")
        device = parts[-3].lower()
        label = os.path.splitext(os.path.basename(filepath))[0].lower()

    return label, device

def visualize_keystrokes(audio, sr, segment_starts, title="Detected Keystrokes"):
    plt.figure(figsize=(12, 3))
    librosa.display.waveshow(audio, sr=sr)
    for start_time in segment_starts:
        plt.axvline(start_time, color='red', linestyle='--')
    plt.title(title)
    plt.show()
    
def add_noise(audio, noise_level=0.005):
    noise = np.random.randn(len(audio))
    augmented = audio + noise_level * noise
    return augmented.astype(np.float32)

def random_gain(audio, min_gain=0.8, max_gain=1.2):
    gain = np.random.uniform(min_gain, max_gain)
    return audio * gain

def pitch_shift(audio, sr, n_steps=1):
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

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

class AudioKeystrokeDataset(Dataset):
    def __init__(self, dataset_path, energy_threshold=100, segment_length=14400,
                 sr=44100, n_fft=1024, hop_length=256, n_mels=64, spec_hop_length=500,
                 target_keystrokes=35, full_dataset=False):
        """
        Parameters:
          dataset_path: Path to the dataset directory containing key .wav files.
          energy_threshold: Initial energy threshold for keystroke isolation.
          segment_length: Number of samples per isolated keystroke.
          sr: Sample rate for audio.
          n_fft: FFT window size for STFT.
          hop_length: Hop length for STFT.
          n_mels: Number of mel bands for spectrogram.
          spec_hop_length: Hop length used in mel spectrogram generation.
          target_keystrokes: Expected number of keystrokes per file.
        """
        self.dataset_path = dataset_path
        self.energy_threshold = energy_threshold
        self.segment_length = segment_length
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.spec_hop_length = spec_hop_length
        self.target_keystrokes = target_keystrokes
        self.full_dataset = full_dataset
        
        # List to store (mel-spectrogram, label) tuples
        self.samples = []
        self._prepare_dataset()
        
        # Build a mapping from string labels to integer indices.
        self.label2idx = {label: idx for idx, label in enumerate(sorted(self.get_labels()))}
        self.device2idx = {device: idx for idx, device in enumerate(sorted(self.get_devices()))}
        
    def _prepare_dataset(self):
        """
        Walk through the dataset path, process each .wav file,
        isolate keystrokes, and generate augmented mel-spectrograms.
        """
        # Gather all .wav file paths first for tqdm progress bar.
        all_files = []
        for root, _, files in os.walk(self.dataset_path):
            for file in files:
                if file.endswith('.wav'):
                    all_files.append(os.path.join(root, file))
                    
        # Process each file with tqdm for progress visualization.
        for file_path in tqdm(all_files, desc="Processing Audio Files"):
            label, device_id = extract_label_and_device(file_path, self.full_dataset)
            try:
                audio, sr = librosa.load(file_path, sr=self.sr)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
            
            # Process each keystroke segment.
            # Apply random time shift for data augmentation.
            augmented_segment = time_shift(audio)
            # Apply other data augmentations.
            if np.random.rand() < 0.5:
                augmented_segment = add_noise(augmented_segment)
            if np.random.rand() < 0.5:
                augmented_segment = random_gain(augmented_segment)
            if np.random.rand() < 0.2:
                augmented_segment = pitch_shift(augmented_segment, sr, n_steps=np.random.uniform(-1, 1))
            # Generate mel-spectrogram and apply SpecAugment.
            mel_spec = generate_mel_spectrogram(
                augmented_segment, sr, n_mels=self.n_mels,
                n_fft=self.n_fft, hop_length=self.spec_hop_length
            )
            mel_spec_aug = spec_augment(mel_spec)
            # Add it to your sample
            self.samples.append((mel_spec_aug, label, device_id))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        mel_spec, label, device = self.samples[idx]
        # Normalize the spectrogram to the range [0, 1]
        mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-6)
        # Map the string label to an integer using the label mapping
        label_idx = self.label2idx[label]
        device_idx = self.device2idx[device]
        # Convert the label to a torch tensor
        label_tensor = torch.tensor(label_idx, dtype=torch.long)
        device_tensor = torch.tensor(device_idx, dtype=torch.long)
        return mel_spec, label_tensor, device_tensor
    
    def get_labels(self):
        return list(set(label for _, label, _ in self.samples))
    
    def get_devices(self):
        return list(set(device for _, _, device in self.samples))