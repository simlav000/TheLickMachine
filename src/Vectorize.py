import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt


def vectorize(fname: str):
    # Load the audio file. Set sample rate to 16kHz
    y, sr = librosa.load(fname, sr=16000)

    # Compute the Mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
        fmin=20,  # Lowest frequency for human hearing
        fmax=8000  # Highest frequency for human hearing
    )

    # Compute delta (1st "derivative") of mel-spectrogram
    mel_spec_delta = librosa.feature.delta(mel_spec)

    # Compute delta-delta (2nd "derivative")
    mel_spec_delta2 = librosa.feature.delta(mel_spec, order=2)

    # Compute the Chromagram
    chromagram = librosa.feature.chroma_stft(
        y=y,
        sr=sr,
        n_fft=2048,
        hop_length=512
    )

    from scipy.ndimage import zoom
    # Resize the chromagram to match the Mel spectrogram's dimensions
    chromagram_resized = zoom(chromagram, (mel_spec.shape[0] / chromagram.shape[0], 1), order=1)  # Linear interpolation


    features = np.stack(
             [
             mel_spec,
             mel_spec_delta,
             mel_spec_delta2,
             chromagram_resized
             ],
             axis=0
    )

    epsilon = 1e-10
    features = (features - np.mean(features, axis=(1, 2), keepdims=True)) / (np.std(features, axis=(1, 2), keepdims=True) + epsilon)

    return features

def plot_features(features, sr=16000):
    mel_spec, mel_spec_delta, mel_spec_delta2, chromagram = features

    # Create a figure with four subplots
    plt.figure(figsize=(12, 16))

    # Plot the Mel-spectrogram
    plt.subplot(4, 1, 1)
    librosa.display.specshow(
        librosa.power_to_db(mel_spec, ref=np.max),
        y_axis='mel',
        x_axis='time',
        sr=sr
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')

    # Plot the Delta of Mel-spectrogram
    plt.subplot(4, 1, 2)
    librosa.display.specshow(
        librosa.power_to_db(mel_spec_delta, ref=np.max),
        y_axis='mel',
        x_axis='time',
        sr=sr
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title('Delta of Mel Spectrogram')

    # Plot the Delta-Delta of Mel-spectrogram
    plt.subplot(4, 1, 3)
    librosa.display.specshow(
        librosa.power_to_db(mel_spec_delta2, ref=np.max),
        y_axis='mel',
        x_axis='time',
        sr=sr
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title('Delta-Delta of Mel Spectrogram')

    # Plot the Chromagram
    plt.subplot(4, 1, 4)
    librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', cmap='coolwarm')
    plt.colorbar(label='Chromagram amplitude')
    plt.title('Chromagram')

    # Show the plots
    plt.tight_layout()
    plt.show()


def check_normalization(features):
    means = np.mean(features, axis=(1, 2))
    stds = np.std(features, axis=(1, 2))
    print("Means per channel:", means)
    print("Standard deviations per channel:", stds)

def process_audio_directory(input_dir, output_file):
    # List to hold feature arrays
    all_features = []

    # Iterate through all files in the directory
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".mp3"):
                file_path = os.path.join(root, file)
                try:
                    features = vectorize(file_path)
                    all_features.append(features)
                except Exception as e:
                    print(f"Failed to process {file_path}: {e}")

    # Stack all features into one numpy array
    dataset = np.stack(all_features, axis=0)
    np.save(output_file, dataset)

    print(f"Dataset saved to {output_file}")
if __name__ == "__main__":
    # Path to the audio file
    sample_lick_fpath = rf"/home/simlav000/McGill/GroupProjects/TheLickMachine/data/TheLick-ALL_2022-02-07v5/sample/"
    process_audio_directory(sample_lick_fpath, "audio_dataset_sample.npy")
