import os
import soundfile
import random
from typing import cast
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt



def vectorize(y, sr: int = 16000) -> list:
    # Compute the Mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=2048,
        hop_length=512,  # number of samples between successive frames
        n_mels=128,
        # Hear for yourself: https://www.szynalski.com/tone-generator/
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

    features_raw = [mel_spec, mel_spec_delta, mel_spec_delta2, chromagram]
    return features_raw

def fix_length(y, target_duration: float = 5.0, sr = 16000):
    # Calculate target length in samples
    target_length = int(target_duration * sr)

    # Trim or pad the audio to the target length
    y = librosa.util.fix_length(y, size=target_length)

    return y


def change_tempo(y):
    tempo_factor = random.uniform(0.5, 1.5)
    # Adjust the tempo
    modified_audio = librosa.effects.time_stretch(y, rate=tempo_factor)

    return modified_audio


def add_pink_noise(y, noise_level=0.1):
    # Generate pink noise with the same length as the audio signal
    white_noise = np.random.normal(0, 1, y.shape)  # Generate white noise
    pink_noise = np.cumsum(white_noise)  # Simple 1/f filtering to create pink noise
    pink_noise -= pink_noise.mean()  # Center the noise around zero
    pink_noise = pink_noise / np.max(np.abs(pink_noise))  # Normalize to [-1, 1]

    # Scale the pink noise to control the volume relative to the original audio
    pink_noise = pink_noise * noise_level

    # Add the pink noise to the original audio
    y_noisy = y + pink_noise

    return y_noisy


def pitch_shift(y, sr=16000, min_shift=-5, max_shift=5):
    # Generate a random pitch shift between min_shift and max_shift
    pitch_shift = random.uniform(min_shift, max_shift)

    y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_shift)

    return y_shifted


def transform(y):
    return pitch_shift(add_pink_noise(fix_length(change_tempo(y))))


def reconstruct_audio(y, sr=16000):
    y_reconstructed = librosa.feature.inverse.mel_to_audio(y, sr=sr, n_iter=32, hop_length=512)

    # Save the reconstructed audio
    soundfile.write('reconstructed_audio.wav', y_reconstructed, sr)


def normalize(features):

    mel_spec, mel_spec_delta, mel_spec_delta2, chromagram = features
    from scipy.ndimage import zoom
    # Resize the chromagram to match the Mel spectrogram's dimensions
    chromagram_resized = zoom(
        # Linear interpolation
        chromagram, (mel_spec.shape[0] / chromagram.shape[0], 1), order=1)

    features = np.stack(
        [mel_spec, mel_spec_delta, mel_spec_delta2, chromagram_resized], axis=0)

    epsilon = 1e-10
    features = (features - np.mean(features, axis=(1, 2), keepdims=True)
                ) / (np.std(features, axis=(1, 2), keepdims=True) + epsilon)

    return features


def process_audio_directory(input_dir, output_file):
    # List to hold feature arrays
    all_features = []

    mp3_files = [f for f in os.listdir(input_dir) if f.endswith(".mp3")]

    for idx, file in enumerate(mp3_files, start=1):
        file_path = os.path.join(input_dir, file)
        try:
            y, _ = librosa.load(file_path, sr=16000)
            features = normalize(vectorize(transform(y)))
            all_features.append(features)
        except Exception as e:
            print(f"Failed to process {file_path}: {e}")

        print(f"Processed {idx}/{len(mp3_files)}: {file}")

    # Stack all features into one numpy array
    dataset = np.stack(all_features, axis=0)
    np.save(output_file, dataset)

    print(f"Dataset saved to {output_file}")


if __name__ == "__main__":
    pwd = os.environ["PWD"]

    dir_positive = os.path.join(pwd, "../data/TheLick_2022-02-11_v6/TheLick_2022-02-07_v5/external/positives/")
    #sample = r"../data/PositiveExternal/lick_0001.mp3"

    process_audio_directory(dir_positive, "external_negatives_transformed_vectorized.npy")
