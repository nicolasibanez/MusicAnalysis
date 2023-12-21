import librosa
import numpy as np
import matplotlib.pyplot as plt

def major_key_template():
    # C Major: C, E, G
    return np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])

def minor_key_template():
    # C Minor: C, Eb, G
    return np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0])


def detect_key(y, sr):
    # Extract Chroma Features
    chroma = librosa.feature.chroma_cens(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    # Key templates
    major = major_key_template()
    minor = minor_key_template()

    # Compute Correlation
    major_correlations = [np.correlate(chroma_mean, np.roll(major, i))[0] for i in range(12)]
    minor_correlations = [np.correlate(chroma_mean, np.roll(minor, i))[0] for i in range(12)]

    # Find Key with Highest Correlation
    max_corr = max(major_correlations + minor_correlations)
    key_idx = np.argmax(major_correlations + minor_correlations)
    if key_idx < 12:
        key = librosa.midi_to_note(key_idx) + " Major"
    else:
        key = librosa.midi_to_note(key_idx - 12) + " Minor"

    return key


def compute_key_correlations(y, sr):
    # Extract Chroma Features
    chroma = librosa.feature.chroma_cens(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    # Key templates
    major = major_key_template()
    minor = minor_key_template()

    # Compute Correlation for all keys
    major_correlations = [np.correlate(chroma_mean, np.roll(major, i))[0] for i in range(12)]
    minor_correlations = [np.correlate(chroma_mean, np.roll(minor, i))[0] for i in range(12)]

    # Combine correlations
    correlations = major_correlations + minor_correlations
    keys = [librosa.midi_to_note(i) + " Major" for i in range(12)] + [librosa.midi_to_note(i) + " Minor" for i in range(12)]

    return keys, correlations

def plot_key_correlations(y, sr):
    keys, correlations = compute_key_correlations(y, sr)

    plt.figure(figsize=(10, 6))
    plt.bar(keys, correlations)
    plt.xlabel('Keys')
    plt.ylabel('Correlation')
    plt.xticks(rotation=45)
    plt.title('Key Correlation with Audio')
    plt.tight_layout()
    plt.show()

def main():
    path = 'podcast001/'
    song = 'Alignment - Future Dancefloor [VNR043  A2].mp3'
    song = 'AMRK - Exhausted (ft. Dr. Pops) [FREE DL].mp3'
    y, sr = librosa.load(path + song, mono=True)


    chroma = librosa.feature.chroma_cens(y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)

    print(chroma.shape)
    print(chroma)

    print(tonnetz.shape)
    print(tonnetz)

    # plot
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
    plt.colorbar()
    plt.title('Chromagram')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()