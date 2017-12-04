from pathlib import Path
from glob import glob
import numpy as np
# np.set_printoptions(threshold=np.inf)

import matplotlib.pyplot as plt
import librosa.display
import librosa
import sys

# base_dir = "./chroma_arrays/"
base_dir = "./test_arrays/"
"""
Prints to stderr
"""
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

""" 
Prints arrays with commas
"""
def print_data(data):
    # The indexing removes the quotes at the start and end
    print(repr(data)[1:-1])

""" 
Creates a chromagram for a file
"""
def chroma(file_name):
    # An 11 kHz sample rate is used because it was suggested in the paper.
    eprint('Processing file: {}'.format(file_name))
    y, sr = librosa.load(file_name, sr=22050)

    # Set the hop length
    hop_length = 512

    # Separate harmonics and percussives into two waveforms
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    # Beat track on the percussive signal
    tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr)
    beat_t = librosa.frames_to_time(beat_frames, sr=sr)

    # Compute chroma features from the harmonic signal
    chromagram = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)

    # Aggregate chroma features between beat events # The median feature is used instead
    beat_chroma = librosa.util.sync(chromagram, beat_frames, aggregate=np.median)

    # commented out for now
    #chromagram, beat_chroma, target_file, target_image_file = write_chroma_info(file_name, chromagram, beat_chroma)
    save_chroma_pics(chromagram, beat_chroma, beat_frames, beat_t, sr, "Please Please Me", "Please Please Me")

    return chromagram, beat_chroma, beat_frames, beat_t, sr
""" 
Writes the chroma and beat arrays
"""
def write_chroma_info(file_name, chromagram, beat_chroma):
    # Extract the file name
    chroma_index = file_name.find('/') + 1
    extension_index = file_name.find('.wav')
    target_file = file_name[chroma_index:extension_index]

    # py doesn't have to be added as an extension
    chroma_array_dir = 'chroma_arrays/'
    target_python_file = chroma_array_dir + target_file

    chroma_image_dir = 'chroma_images/'
    target_image_file = chroma_image_dir + target_file + '.png'

    # eprint('chromogram (to file): {}'.format(chromagram))
    target_chroma_file = target_python_file + '_chroma'
    np.save(target_chroma_file, chromagram)

    # eprint('beat_chroma (to file): {}'.format(beat_chroma))
    target_beat_file = target_python_file + '_beat'
    np.save(target_beat_file, beat_chroma)
    return chromagram, beat_chroma, target_file, target_image_file

"""
Saves the chroma pictures
"""
def save_chroma_pics(chromagram, beat_chroma, beat_frames, beat_t, sr, target_file, target_image_file):


    ax1 = plt.subplot(2, 1, 1)
    librosa.display.specshow(chromagram, y_axis='chroma', x_axis='time')
    plt.title('Chroma (linear time)')

    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    librosa.display.specshow(beat_chroma, y_axis='chroma', x_axis='time', x_coords=beat_t)
    plt.title('Chroma (beat time)')
    # plt.tight_layout()

    plt.suptitle(target_file)
    # plt.colorbar()
    plt.savefig(target_image_file)
    eprint('Done Processing: {}'.format(target_file))
    eprint('')

def file_exists(file_name):
    song_title_cutoff = 11
    base_file_name = file_name[song_title_cutoff:file_name.find('.wav')]
    saved_file_name = base_file_name + "_beat.npy"
    chroma_array_path = base_dir + saved_file_name
    my_file = Path(chroma_array_path)
    # eprint('chroma_array_path: {}'.format(chroma_array_path))
    file_exists = my_file.exists()
    # eprint('file_exists: {}'.format(file_exists))
    return file_exists

"""
Rewrite chroma images
"""
def rewrite_chroma_images():
    list_of_wavs = glob('song_files/*wav')
    sr = 11000
    for f in list_of_wavs:
        chroma_index       = file_name.find('/') + 1
        file_extension = '.wav'
        extension_index    = file_name.find(file_extension)

        target_file = file_name[chroma_index:extension_index]

        chromagram      = np.load()
        beat_chromagram = np.load()
        # write_chroma_info()
            
        

""" 
Processes all the song files in the directory that contains the song files.
"""
def main():
    list_of_wavs = glob('song_files/*wav')
    eprint('list_of_wavs: {}'.format(len(list_of_wavs)))
    i = 0
    for f in list_of_wavs:
        # i += 1
        # if i < start:
        #     continue
        if not file_exists(f):
            chroma(f)
        # chroma(f)

if __name__ == '__main__':
    main()
