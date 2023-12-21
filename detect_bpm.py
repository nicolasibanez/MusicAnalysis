import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.fft import fft
from mutagen.id3 import ID3
from mutagen.mp3 import MP3
from mutagen.id3 import TBPM
from tqdm import tqdm

import os

# Ensure librosa's display module is imported
import librosa.display

def load_mp3(path):
    y, sr = librosa.load(path, mono=True)
    return y, sr

def compute_autocorrelation(audio, sr, bpm_range):
    max_correlation = 0
    optimal_delta_time = None
    for bpm in bpm_range:
        delta_time = 60/bpm
        shift = int(delta_time * sr)  # Convert to samples
        correlation = np.correlate(audio[:-shift], audio[shift:])
        if correlation > max_correlation:
            max_correlation = correlation
            optimal_delta_time = delta_time
    return int(60/optimal_delta_time)

def plot_autocorrelation(audio, sr, bpm_range):
    correlation = []
    for bpm in bpm_range:
        delta_time = 60/bpm
        shift = int(delta_time * sr)  # Convert to samples
        correlation.append(np.correlate(audio[:-shift], audio[shift:]))
    plt.plot(bpm_range, correlation)
    plt.show()


def main():
    path_podcast = 'podcast001/'
    bpm_range = range(120, 191)

    # song = 'Jaëss - Kick Dans Ton Cor.mp3'
    # y, sr = load_mp3(path_podcast + song)

    # plot_autocorrelation(y, sr, bpm_range)

    # # plot the waveform
    # plt.figure(figsize=(14, 5))
    
    # plt.plot(y)
    # plt.show()

    # Get all .mp3 files from the directory
    all_files = [f for f in os.listdir(path_podcast) if f.endswith(".mp3")]

    # Wrap the loop with tqdm for a progress bar
    for filename in tqdm(all_files, desc="Processing files"):
        path = path_podcast + filename
        y, sr = load_mp3(path)
        bpm = compute_autocorrelation(y, sr, bpm_range)

        # Set BPM in metadata
        audio = MP3(path)
        audio.tags.add(TBPM(encoding=3, text=str(bpm)))  # Correct way to set BPM
        audio.save()

        # Update tqdm's progress bar description with the file and BPM info
        tqdm.write(f'File: {filename} - BPM: {bpm}')


if __name__ == "__main__":
    main()
