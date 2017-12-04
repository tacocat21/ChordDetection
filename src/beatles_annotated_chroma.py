import os
import numpy as np
import librosa as librosa
import rosa_chroma as ish_chroma
import ipdb
import util
import json


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
    labels = []
    for line in range(len(chordlab)):
        c = chordlab[line]

        # commenting these out because silence can still be part of a song
        #if c[2] == 'N':
        #    continue
        labels.append(c[2])
        start   = float(c[0])
        end     = float(c[1])

        chroma_range = np.searchsorted(chroma_times, [start, end], side='left')
        print('Line: {} - [{} {}] = chromas: [{} {}]'.format(line, start, end, chroma_range[0], chroma_range[1]))
        chromas = chromagram[:, chroma_range[0]:chroma_range[1]]
        anno_chroma.append(chromas.tolist())

    return np.array(anno_chroma), labels

def read_chordlab(chord_file):
    # Read in the space-delimited chord label file
    return [line.rstrip('\n').split(' ') for line in open(chord_file)]

def map_beatles_dataset():
    """
    Map the whole beatles data set and save the output to a JSON
    :return: 
    """
    res = {}
    hopsize= 512
    chord_ext = ['.lab']
    audio_ext = ['.mp3', '.wav', '.flac']
    song_folders = os.listdir(util.BEATLES_SONG)
    chord_folders = os.listdir(util.BEATLES_CHORD)
    chromagrams = []
    annotated_chromagram = []
    label_list = []
    for song_f in song_folders:
        if song_f not in chord_folders:
            continue
        print('Found: {}'.format(song_f))
        song_files = os.listdir(os.path.join(util.BEATLES_SONG, song_f))
        chord_files = os.listdir(os.path.join(util.BEATLES_CHORD, song_f))
        # assuming that song order for chord and songs are the same
        song_files = remove_song_without_ext(song_files, audio_ext)
        chord_files = remove_song_without_ext(chord_files, chord_ext)
        assert(len(song_files) == len(chord_files))
        for i in range(len(song_files)):
            chord_lab = read_chordlab(os.path.join(util.BEATLES_CHORD, song_f, chord_files[i]))
            chromagram, beat_chroma, beat_frames, beat_t, sr = ish_chroma.chroma(os.path.join(util.BEATLES_SONG, song_f, song_files[i]))
            anno_chromas, labels = map_chroma(chromagram, sr, hopsize, chord_lab)
            chromagrams.append(chromagram.tolist())
            annotated_chromagram.append(anno_chromas.tolist())
            label_list.append(labels)

    res['chromagram'] = np.array(chromagrams)
    res['annotated_chromas'] = np.array(annotated_chromagram)
    res['labels'] = label_list
    return res

def save_json(file_name, json_dict):
    with open(file_name, 'w') as fp:
        json.dump(jsonify(json_dict), fp)

def jsonify(data):
    # code from https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
    json_data = dict()
    for key, value in data.items():
        if isinstance(value, list):  # for lists
            value = [jsonify(item) if isinstance(item, dict) else item for item in value]
        if isinstance(value, dict):  # for nested lists
            value = jsonify(value)
        if isinstance(key, int):  # if key is integer: > to string
            key = str(key)
        if type(value).__module__ == 'numpy':  # if value is numpy.*: > to python list
            value = value.tolist()
        json_data[key] = value
    return json_data

def remove_song_without_ext(files, ext):
    res = []
    for f in files:
        for e in ext:
            if f.endswith(e):
                res.append(f)
    res.sort()
    return res

if __name__ == "__main__":
    """
    ipdb.set_trace()
    song_dir = util.BEATLES_DIR
    chord_dir = util.BEATLES_CHORD
    album = '01_-_Please_Please_Me/'
    title = '07_-_Please_Please_Me'
    audio_ext = '.mp3'
    chord_ext = '.lab'
    hopsize = 512

    song = song_dir + '/' + title + audio_ext # should include 'album' but my file structure is not organized that way yet
    chords = chord_dir + '/' + album + title + chord_ext # like this

    song_file = os.path.realpath(os.path.join(os.path.dirname(__file__), song))
    chord_file = os.path.realpath(os.path.join(os.path.dirname(__file__), chords))

    # Read in the space-delimited chord label file
    chordlab = read_chordlab(chord_file)

    # Run Ish's program to get the chroma information
    chromagram, beat_chroma, beat_frames, beat_t, sr = ish_chroma.chroma(song_file)

    # Find corresponding frames
    anno_chromas, labels = map_chroma(chromagram, sr, hopsize, chordlab)
    for a in anno_chromas:
        print(a.shape)
    """
    ipdb.set_trace()
    res = map_beatles_dataset()
    #jsonify(res)
    save_json('beatle_data.json', res)




