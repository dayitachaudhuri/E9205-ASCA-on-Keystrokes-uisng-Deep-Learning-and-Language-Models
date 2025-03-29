import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

DATASET_PATH = "../data/Zoom/Raw Data"

# Utility functions (as provided)
def extract_label(filepath):
    # Extracts the key label from the file name (e.g. "1.wav" -> "1")
    return os.path.splitext(os.path.basename(filepath))[0]

def isolate_keystrokes(audio, sr, threshold, segment_length=14400, n_fft=1024, hop_length=256):
    """
    Isolate keystrokes from audio using STFT energy thresholding.
    """
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    energy = np.sum(np.abs(stft), axis=0)
    keystroke_frames = np.where(energy > threshold)[0]
    groups = np.split(keystroke_frames, np.where(np.diff(keystroke_frames) > 1)[0] + 1)
    
    segments = []
    for group in groups:
        center_frame = int(np.mean(group))
        start = max((center_frame * hop_length) - segment_length // 2, 0)
        end = start + segment_length
        if end > len(audio):
            start = len(audio) - segment_length
            end = len(audio)
        segments.append(audio[start:end])
    return segments

def zoom_keystroke_threshold(audio, sr, initial_threshold=100, step=5, target_keystrokes=25):
    """
    Zoom threshold algorithm to automatically adjust the energy threshold such that
    exactly `target_keystrokes` segments are isolated.
    
    Returns:
      optimal_threshold: The threshold value that produced target_keystrokes.
      segments: The list of isolated keystroke segments.
    """
    P = initial_threshold
    while True:
        segments = isolate_keystrokes(audio, sr, threshold=P)
        num_keystrokes = len(segments)
        if num_keystrokes == target_keystrokes:
            break
        elif num_keystrokes < target_keystrokes:
            P -= step
        elif num_keystrokes > target_keystrokes:
            P += step
        step *= 0.99 
    return P, segments

def time_shift(segment, max_shift_percentage=0.4):
    """
    Randomly time-shift an audio segment for data augmentation.
    """
    max_shift = int(len(segment) * max_shift_percentage)
    shift = np.random.randint(-max_shift, max_shift)
    shifted_segment = np.roll(segment, shift)
    if shift > 0:
        shifted_segment[:shift] = 0
    elif shift < 0:
        shifted_segment[shift:] = 0
    return shifted_segment

def generate_mel_spectrogram(segment, sr, n_mels=64, n_fft=1024, hop_length=225):
    """
    Convert an audio segment into a mel-spectrogram (in dB scale).
    """
    S = librosa.feature.melspectrogram(y=segment, sr=sr, n_fft=n_fft,
                                       hop_length=hop_length, n_mels=n_mels)
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB

def spec_augment(spectrogram, time_mask_percentage=0.1, freq_mask_percentage=0.1):
    """
    Apply SpecAugment masking to a mel-spectrogram.
    """
    augmented_spec = spectrogram.copy()
    n_mels, n_time = augmented_spec.shape
    time_mask_length = max(1, int(n_time * time_mask_percentage))
    freq_mask_length = max(1, int(n_mels * freq_mask_percentage))
    time_start = np.random.randint(0, n_time - time_mask_length + 1)
    freq_start = np.random.randint(0, n_mels - freq_mask_length + 1)
    mask_value = np.mean(augmented_spec)
    augmented_spec[freq_start:freq_start + freq_mask_length,
                   time_start:time_start + time_mask_length] = mask_value
    return augmented_spec

# The Audio Dataset Class
class AudioKeystrokeDataset(Dataset):
    def __init__(self, dataset_path, energy_threshold=100, segment_length=14400,
                 sr=44100, n_fft=1024, hop_length=256, n_mels=64, spec_hop_length=225,
                 target_keystrokes=25):
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
        
        # List to store (mel-spectrogram, label) tuples
        self.samples = []
        self._prepare_dataset()

    def _prepare_dataset(self):
        """
        Walk through the dataset path, process each .wav file,
        isolate keystrokes, and generate augmented mel-spectrograms.
        """
        for root, _, files in os.walk(self.dataset_path):
            for file in files:
                if file.endswith('.wav'):
                    file_path = os.path.join(root, file)
                    label = extract_label(file_path)
                    try:
                        audio, sr = librosa.load(file_path, sr=self.sr)
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
                        continue
                    
                    # Compute optimal threshold and isolate keystrokes from the file.
                    optimal_threshold, segments = zoom_keystroke_threshold(
                        audio, sr, initial_threshold=self.energy_threshold,
                        target_keystrokes=self.target_keystrokes
                    )
                    
                    # Process each keystroke segment.
                    for segment in segments:
                        # Apply random time shift for data augmentation.
                        augmented_segment = time_shift(segment)
                        # Generate mel-spectrogram and apply SpecAugment.
                        mel_spec = generate_mel_spectrogram(
                            augmented_segment, sr, n_mels=self.n_mels,
                            n_fft=self.n_fft, hop_length=self.spec_hop_length
                        )
                        mel_spec_aug = spec_augment(mel_spec)
                        # Save the (spectrogram, label) pair.
                        self.samples.append((mel_spec_aug, label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Retrieve the augmented mel-spectrogram and corresponding label for a given index.
        """
        mel_spec, label = self.samples[idx]
        # Normalize the spectrogram to the range [0, 1]
        mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-6)
        # Optionally convert label to integer if keys are represented as numbers
        try:
            label = int(label)
        except ValueError:
            pass
        return mel_spec, label

if __name__ == '__main__':
    DATASET_PATH = "../data/Zoom/Raw Data"
    dataset = AudioKeystrokeDataset(DATASET_PATH)
    print(f"Dataset contains {len(dataset)} keystroke samples.")

    spec, lbl = dataset[0]
    plt.figure(figsize=(4, 4))
    librosa.display.specshow(spec, sr=44100, x_axis='time', y_axis='mel', cmap='viridis')
    plt.title(f"Label: {lbl}")
    plt.colorbar(format='%+2.0f dB')
    plt.show()
