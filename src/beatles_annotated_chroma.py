import os
import numpy as np
import librosa as librosa
import rosa_chroma as ish_chroma



"""
chromagram - 12xN. A chroma per frame (corr to spectrogram). N > M
beat_chroma - 12xM - Percussive chromagram for each 'downbeat'
beat_frames - 1xM - 1D array of indices for 'chomagram' where the beats occur
beat_t - 1xM - 1D array of times (sec) that the beats (chroma and frames) occur
"""

# Use chromagram, sample rate, hopsize, and chordlabel file to gather corresponding chromagrams
# for each line of the chordlabel file.
# Returns an Lx(12xM), where M is variable length dependent on the duration specified in chordlab[L]
def map_chroma(chromagram, sr, hopsize, chordlab):
    chroma_times = librosa.core.frames_to_time(np.arange(chromagram.shape[1] + 1), sr=sr, hop_length=hopsize)

    anno_chroma = []
    for line in range(len(chordlab)):
        c = chordlab[line]

        if c[2] == 'N':
            continue

        start   = float(c[0])
        end     = float(c[1])

        chroma_range = np.searchsorted(chroma_times, [start, end], side='left')
        print('Line: {} - [{} {}] = chromas: [{} {}]'.format(line, start, end, chroma_range[0], chroma_range[1]))
        chromas = chromagram[:, chroma_range[0]:chroma_range[1]]
        anno_chroma.append(chromas)

    return anno_chroma

song_dir = '..\\data\\The_Beatles_Annotations\\'
chord_dir = '..\\data\\The_Beatles_Annotations\\chordlab\\The_Beatles\\'
album = '01_-_Please_Please_Me\\'
title = '07_-_Please_Please_Me'
audio_ext = '.mp3'
chord_ext = '.lab'
hopsize = 512

song = song_dir + title + audio_ext # should include 'album' but my file structure is not organized that way yet
chords = chord_dir + album + title + chord_ext # like this

song_file = os.path.realpath(os.path.join(os.path.dirname(__file__), song))
chord_file = os.path.realpath(os.path.join(os.path.dirname(__file__), chords))

# Read in the space-delimited chord label file
chordlab = [line.rstrip('\n').split(' ') for line in open(chord_file)]

# Run Ish's program to get the chroma information
chromagram, beat_chroma, beat_frames, beat_t, sr = ish_chroma.chroma(song_file)

# Find corresponding frames
anno_chromas = map_chroma(chromagram, sr, hopsize, chordlab)
for a in anno_chromas:
    print(a.shape)



