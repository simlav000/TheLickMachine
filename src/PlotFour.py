import librosa
import matplotlib.pyplot as plt
import numpy as np

# Plot spectogram
def plot_four(data, sr=16000, name="temp"):
    # Create a figure with four subplots
    plt.figure(figsize=(12, 16))

    # Plot the Mel-spectrogram
    plt.subplot(4, 1, 1)
    mel_spec = data[0]
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
    mel_spec_delta = data[1]
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
    mel_spec_delta2=data[2]
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
    chromagram=data[3]
    librosa.display.specshow(chromagram, x_axis='time',
                             y_axis='chroma', cmap='coolwarm')
    plt.colorbar(label='Chromagram amplitude')
    plt.title('Chromagram')

    # Show the plots
    plt.tight_layout()
    plt.savefig(name + ".jpg")
