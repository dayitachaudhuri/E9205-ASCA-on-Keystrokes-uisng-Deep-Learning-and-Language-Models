import os
import random
import librosa
import soundfile as sf
from tqdm import tqdm
import numpy as np
import json

DEVICES = ['mac', 'lenovo', 'msi', 'messnger', 'zoom']
CHAR_MAPPINGS = {
    '.': ['fullstop'],
    ',': ['comma(,)'],
    "'": ["apostrophe(')"],
    '/': ['slash'],
    '\\': ['backslash'],
    '?': ['Lshift', 'slash'],
    ';': ['semicolon(;)'],
    ':': ['Lshift', 'semicolon(;)'],
    '-' : ['dash(-)'],
    '_': ['Lshift', 'dash(-)'],
    '=': ['equal(=)'],
    '+': ['Lshift', 'equal(=)'],
    ')': ['Lshift', '0'],
    '!': ['Lshift', '1'],
    '@': ['Lshift', '2'],
    '#': ['Lshift', '3'],
    '$': ['Lshift', '4'],
    '%': ['Lshift', '5'],
    '^': ['Lshift', '6'],
    '&': ['Lshift', '7'],
    '*': ['Lshift', '8'],
    '(': ['Lshift', '9'],
    '{': ['Lshift', 'bracketopen([)'],
    '}': ['Lshift', 'bracketclose(])'],
    '[': ['bracketopen([)'],
    ']': ['bracketclose(])'],
    ' ': ['space'],
    '"' : ['Lshift', "apostrophe(')"],
}

def load_keystroke_paths(base_dir='data/ProcessedWithImage'):
    keystrokes = {}
    for key in os.listdir(base_dir):
        key_path = os.path.join(base_dir, key)
        if not os.path.isdir(key_path):
            continue
        for entry in os.listdir(key_path):
            for device in DEVICES:
                if entry.endswith(device) and entry.startswith(key):
                    if key not in keystrokes:
                        keystrokes[key] = {}
                    device_path = os.path.join(key_path, entry)
                    files = [os.path.join(device_path, f) for f in os.listdir(device_path) if f.endswith('.wav')]
                    if files:
                        keystrokes[key][device] = files
                    break
    return keystrokes

# Update the function definition of sentence_to_keystrokes
def sentence_to_keystrokes(sentence, keystrokes, device=None):
    sentence = sentence.strip()
    audio_sequence = []
    if device is None:
        device = random.choice(DEVICES)
    key_sequence = []
    for char in sentence:
        print(f"Character: '{char}'", end=' -> ')
        if char in CHAR_MAPPINGS:
            mapped_keys = CHAR_MAPPINGS[char]
            print(f"Mapped to: {mapped_keys}")
            key_sequence.extend(mapped_keys)
        else:
            lower_char = char.lower()
            print(f"Mapped to: {lower_char}")
            key_sequence.append(lower_char)

    for key in key_sequence:
        if key not in keystrokes:
            print(f"Key '{key}' not found in keystrokes data.")
            continue
        if device not in keystrokes[key]:
            print(f"Device '{device}' not found for key '{key}'.")
            continue

        file = random.choice(keystrokes[key][device])
        print(f"Using file for key '{key}': {file}")
        y, sr = librosa.load(file, sr=None)
        audio_sequence.append(y)

        silence = np.zeros(int(0.2 * sr)) 
        audio_sequence.append(silence)

    if not audio_sequence:
        return None, None

    return np.concatenate(audio_sequence), sr

def process_sentences(sentences_file, output_dir='data/sentences'):
    os.makedirs(output_dir, exist_ok=True)
    keystrokes = load_keystroke_paths()
    sentence2device = {}

    with open(sentences_file, 'r') as f:
        lines = f.readlines()

    for line in tqdm(lines):
        try:
            sid, sentence = line.strip().split('. ', 1)
            sentence = sentence.strip()
            output_path = os.path.join(output_dir, f"{sid}.wav")

            device = random.choice(DEVICES)
            audio, sr = sentence_to_keystrokes(sentence, keystrokes, device=device)

            if audio is not None:
                sf.write(output_path, audio, sr)
                sentence2device[sid] = device
            else:
                print(f"Skipping {sid}: No audio generated")
        except Exception as e:
            print(f"Error with line '{line}': {e}")

    # Save mapping of sentence ID to device
    with open(os.path.join(output_dir, 'sentence2device.json'), 'w') as f:
        json.dump(sentence2device, f, indent=2)

if __name__ == "__main__":
    sentences_file = 'data/5_sentences.txt'
    output_dir = 'data/sentences'
    process_sentences(sentences_file, output_dir)