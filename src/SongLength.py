import os
import numpy as np
import matplotlib.pyplot as plt
from mutagen.mp3 import MP3


def duration(fpath):
    try:
        audio = MP3(fpath)
        return audio.info.length  # Duration in seconds
    except Exception as e:
        print(f"Failed to process {fpath}: {e}")
        return None


def get_song_lengths(input_dir):
    mp3_files = [f for f in os.listdir(input_dir) if f.endswith(".mp3")]

    lengths = []
    for idx, file in enumerate(mp3_files, start=1):
        file_path = os.path.join(input_dir, file)
        length = duration(file_path)
        if length is not None:
            lengths.append(length)

        print(f"Processed {idx}/{len(mp3_files)}: {file}")

    return lengths


if __name__ == "__main__":
    # Replace this with your directory path
    pwd = os.environ["PWD"]
    dir_positive = os.path.join(
        pwd, "../data/TheLick-ALL_2022-02-07v5/positive/")
    dir_negative = os.path.join(
        pwd, "../data/TheLick-ALL_2022-02-07v5/negative/")

    lengths_positive = get_song_lengths(dir_positive)
    lengths_negative = get_song_lengths(dir_negative)

    lengths = np.array(lengths_positive + lengths_negative)

    # Plot histogram
    plt.hist(lengths, bins=20)
    plt.xlabel("Duration (seconds)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Song Lengths")
    plt.savefig("SongLengthDistribution.png")
    plt.show()
