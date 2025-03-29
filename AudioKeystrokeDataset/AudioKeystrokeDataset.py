import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
from tqdm import tqdm 
from torch.utils.data import Dataset

def extract_label(filepath):
    label = os.path.basename(filepath)
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

def isolate_keystrokes(audio, sr, threshold, segment_length=14400, n_fft=1024, hop_length=256):
    """
    Isolate keystrokes from audio using Short-Time Fourier Transform (STFT).
    Parameters:
    - audio: The audio signal to process.
    - sr: Sample rate of the audio.
    - threshold: Energy threshold for detecting keystrokes.
    - segment_length: Length of the segment to extract around each keystroke.
    - n_fft: Number of FFT components.
    - hop_length: Number of samples between frames.
    Returns:
    - segments: List of isolated keystroke segments.
    """
    # Compute the Fast Fourier Transform
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    # Sum the magnitude of the STFT across frequency bins to get energy
    energy = np.sum(np.abs(stft), axis=0)
    # Choose frames based on threashold
    keystroke_frames = np.where(energy > threshold)[0]

    # Group contiguous frames
    groups = np.split(keystroke_frames, np.where(np.diff(keystroke_frames) > 1)[0] + 1)
    
    # Take keystroke as center. Extract segment centered on the keystroke
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
    Implements Algorithm 1: Zoom Keystroke Threshold Setting.
    
    Parameters:
    - audio: 1D numpy array of the audio signal.
    - sr: Sample rate.
    - initial_threshold: Initial energy threshold.
    - step: Step size for adjusting threshold.
    - target_keystrokes: Desired number of keystrokes.
    
    Returns:
    - Optimal threshold value.
    """
    P = initial_threshold
    iterations = 100
    while True and iterations > 0:
        iterations -= 1
        S = isolate_keystrokes(audio, sr, threshold=P)
        num_keystrokes = len(S)
        if num_keystrokes == target_keystrokes:
            break
        elif num_keystrokes < target_keystrokes:
            P -= step
        elif num_keystrokes > target_keystrokes:
            P += step
        step *= 0.99 
    return P, S

def time_shift(segment, max_shift_percentage=0.4):
    """
    Shift the audio segment in time by a random amount.
    Parameters:
    - segment: The audio segment to shift.
    - max_shift_percentage: Maximum percentage of the segment length to shift.
    Returns:
    - shifted_segment: The time-shifted audio segment.
    """
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
    """
    Generate a Mel spectrogram from the audio segment.
    Parameters:
    - segment: The audio segment to process.
    - sr: Sample rate of the audio.
    - n_mels: Number of Mel bands to generate.
    - n_fft: Number of FFT components.
    - hop_length: Number of samples between frames.
    Returns:
    - S_dB: The Mel spectrogram in decibels.
    """
    S = librosa.feature.melspectrogram(y=segment, sr=sr, n_fft=n_fft,  hop_length=hop_length, n_mels=n_mels)
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB

def spec_augment(spectrogram, time_mask_percentage=0.1, freq_mask_percentage=0.1):
    """
    Augment the Mel spectrogram.
    Parameters:
    - spectrogram: The Mel spectrogram to augment.
    - time_mask_percentage: Percentage of the time dimension to mask.
    - freq_mask_percentage: Percentage of the frequency dimension to mask.
    Returns:
    - augmented_spec: The augmented Mel spectrogram.
    """
    augmented_spec = spectrogram.copy()
    n_mels, n_time = augmented_spec.shape
    # Calculate the lengths of the masks
    time_mask_length = max(1, int(n_time * time_mask_percentage))
    freq_mask_length = max(1, int(n_mels * freq_mask_percentage))
    # Randomly select the starting points for the masks
    time_start = np.random.randint(0, n_time - time_mask_length + 1)
    freq_start = np.random.randint(0, n_mels - freq_mask_length + 1)
    # Calculate the mean value of the spectrogram to use as the mask value
    mask_value = np.mean(augmented_spec)
    # Apply the time mask and frequency mask
    augmented_spec[freq_start:freq_start + freq_mask_length, time_start:time_start + time_mask_length] = mask_value
    return augmented_spec

def process_audio_file(file_path, energy_threshold, output_dir, segment_length=14400, sr=44100):
    """
    Process a single audio file to isolate keystrokes and save the segments.
    Parameters:
    - file_path: Path to the audio file.
    - energy_threshold: Energy threshold for isolating keystrokes.
    - output_dir: Directory to save the processed segments.
    - segment_length: Length of the segment to extract around each keystroke.
    - sr: Sample rate for loading the audio.
    Returns:
    - sample_paths: List of paths to the saved audio segments.
    """
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Load the audio file
    audio, sr = librosa.load(file_path, sr=sr)
    segments = isolate_keystrokes(audio, sr, threshold=energy_threshold, segment_length=segment_length)
    
    sample_paths = []
    for idx, segment in enumerate(segments):
        # Apply time shift for data augmentation
        augmented_segment = time_shift(segment)
        # Save the raw audio segment (can be used for further processing or as input to a model)
        segment_path = os.path.join(output_dir, f"keystroke_{idx}.wav")
        sf.write(segment_path, augmented_segment, sr)
        sample_paths.append(segment_path)
        
        # Optionally, generate and save the mel-spectrogram image
        mel_spec = generate_mel_spectrogram(augmented_segment, sr)
        mel_spec_aug = spec_augment(mel_spec)
        plt.figure(figsize=(4, 4))
        librosa.display.specshow(mel_spec_aug, sr=sr, x_axis='time', y_axis='mel', cmap='viridis')
        plt.axis('off')
        image_path = os.path.join(output_dir, f"keystroke_{idx}.png")
        plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    return sample_paths

class AudioKeystrokeDataset(Dataset):
    def __init__(self, dataset_path, energy_threshold=100, segment_length=14400,
                 sr=44100, n_fft=1024, hop_length=256, n_mels=64, spec_hop_length=500,
                 target_keystrokes=35):
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
        # Gather all .wav file paths first for tqdm progress bar.
        all_files = []
        for root, _, files in os.walk(self.dataset_path):
            for file in files:
                if file.endswith('.wav'):
                    all_files.append(os.path.join(root, file))
                    
        # Process each file with tqdm for progress visualization.
        for file_path in tqdm(all_files, desc="Processing Audio Files"):
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
                self.samples.append((mel_spec_aug, label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        mel_spec, label = self.samples[idx]
        # Normalize the spectrogram to the range [0, 1]
        mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-6)
        # Optionally convert label to integer if keys are represented as numbers
        try:
            label = int(label)
        except ValueError:
            pass
        return mel_spec, label