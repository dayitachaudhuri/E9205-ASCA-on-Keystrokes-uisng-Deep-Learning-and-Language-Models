import os
import random
import librosa
import soundfile as sf
from tqdm import tqdm

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
}

def load_keystroke_paths(base_dir='data/Processed'):
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

def sentence_to_keystrokes(sentence, keystrokes):
    sentence = sentence.strip()
    audio_sequence = []
    device = random.choice(DEVICES)
    key_sequence = []
    for char in sentence:
        if char.isupper():
            key_sequence.append('caps')  
            key_sequence.append(char.lower())
            key_sequence.append('caps')
            continue
        elif char in CHAR_MAPPINGS:
            key_sequence.extend(CHAR_MAPPINGS[char])
        else:
            key_sequence.append(char)
    for key in key_sequence:
        if key not in keystrokes or device not in keystrokes[key]:
            print(f"Skipping key '{key}' for device '{device}'")
            continue
        file = random.choice(keystrokes[key][device])
        y, sr = librosa.load(file, sr=None)
        audio_sequence.append(y)
    if not audio_sequence:
        return None, None
    return sum(audio_sequence), sr

def process_sentences(sentences_file, output_dir='data/sentences'):
    os.makedirs(output_dir, exist_ok=True)
    keystrokes = load_keystroke_paths()
    with open(sentences_file, 'r') as f:
        lines = f.readlines()
    for line in tqdm(lines):
        try:
            sid, sentence = line.strip().split('.', 1)
            sentence = sentence.strip()
            output_path = os.path.join(output_dir, f"{sid}.wav")
            audio, sr = sentence_to_keystrokes(sentence, keystrokes)
            if audio is not None:
                sf.write(output_path, audio, sr)
            else:
                print(f"Skipping {sid}: No audio generated")
        except Exception as e:
            print(f"Error with line '{line}': {e}")

if __name__ == "__main__":
    sentences_file = 'data/1000_sentences.txt'
    output_dir = 'data/sentences'
    process_sentences(sentences_file, output_dir)