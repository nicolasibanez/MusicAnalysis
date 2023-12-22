import numpy as np
import pandas as pd
import librosa
import mutagen
import os
from tqdm import tqdm

def estimate_key(chroma):
    # Summarize the chroma features into one vector (average across time)
    chroma_summary = np.mean(chroma, axis=1)

    # Find the index of the maximum value in chroma_summary
    tonic_index = np.argmax(chroma_summary)

    # Major and minor third intervals (4 semitones and 3 semitones respectively)
    major_third_index = (tonic_index + 4) % 12
    minor_third_index = (tonic_index + 3) % 12

    # Compare the strength of the major and minor third
    is_major = chroma_summary[major_third_index] > chroma_summary[minor_third_index]

    # Map index to key (simplified mapping, assuming C Major/A minor are index 0)
    major_keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    minor_keys = ['Am', 'A#m', 'Bm', 'Cm', 'C#m', 'Dm', 'D#m', 'Em', 'Fm', 'F#m', 'Gm', 'G#m']

    estimated_key = major_keys[tonic_index] if is_major else minor_keys[tonic_index]

    return estimated_key

def extract_features(file):
    y, sr = librosa.load(file, mono=True)
    
    # bpm from metadata : 
    audio = mutagen.File(file)
    bpm = audio.tags.get('TBPM').text[0]
    bpm = int(bpm)

    # Energy
    energy = np.sum(np.power(y, 2)) / len(y)
    # print(f'Energy: {energy}')

    # Chroma feature extraction
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    # print(f'Chroma mean shape: {chroma_mean.shape}')

    # Estimate the key
    key = estimate_key(chroma)

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    mfcc_mean = np.mean(mfcc, axis=1)
    # print(f'MFCC mean shape: {mfcc_mean.shape}')

    # Spectral Centroid
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_centroids_mean = np.mean(spectral_centroids, axis=1)
    # print(f'Spectral Centroid mean shape: {spectral_centroids_mean.shape}')

    return [bpm, energy, key] + list(chroma_mean) + list(mfcc_mean) + list(spectral_centroids_mean)


def main():
    path = 'podcast001/'
    songs = [f for f in os.listdir(path) if f.endswith('.mp3')]

    # print(songs)

    print('Extracting Features...')

    features_names = ['song', 'bpm', 'energy', 'key'] + ['chroma_' + str(i) for i in range(12)] + \
                     ['mfcc_' + str(i) for i in range(20)] + \
                     ['spectral_centroids']
    

    df = pd.DataFrame(columns=features_names)

    # Wrap the loop with tqdm for a progress bar
    for song in tqdm(songs, desc="Extracting features"):
        file = path + song
        features = extract_features(file)
        tqdm.write(f'Extracted features for {song}')

        if len(features) + 1 == len(df.columns):  # +1 for the song name
            new_row = pd.DataFrame([pd.Series([song] + features, index=df.columns)])
            if df.empty:
                df = new_row
            else:
                df = pd.concat([df, new_row], ignore_index=True)
        else:
            tqdm.write(f"Error in feature extraction for {song}")
            tqdm.write(f"Expected {len(df.columns)} features, got {len(features) + 1}")  # +1 for song name

    print('Feature extraction completed!')
    print(df.head())

    # Save DataFrame to CSV
    df.to_csv('features.csv', index=False)


if __name__ == '__main__':
    main()