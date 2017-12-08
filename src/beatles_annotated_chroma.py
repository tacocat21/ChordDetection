import json
import os

import ipdb
import librosa as librosa
import numpy as np

import rosa_chroma as ish_chroma
import util
from util import jsonify
import json
import librosa.display
import matplotlib.pyplot as plt

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
        #print('Line: {} - [{} {}] = chromas: [{} {}]'.format(line, start, end, chroma_range[0], chroma_range[1]))
        chromas = chromagram[:, chroma_range[0]:chroma_range[1]]
        anno_chroma.append(chromas.tolist())
    # TODO: fix the missing frames problem
    sum_val = sum([len(n[0]) for n in anno_chroma])
    assert(sum_val <= chromagram.shape[1] *1.01 and sum_val >= chromagram.shape[1]*0.99)
    return anno_chroma, labels

def read_chordlab(chord_file):
    # Read in the space-delimited chord label file
    return [line.rstrip('\n').split(' ') for line in open(chord_file)]

def map_beatles_dataset(hopsize=512, type_='cqt', tol=0.0):
    """
    Map the whole beatles data set and save the output to a JSON
    :return: 
    """
    res = {}
    chord_ext = ['.lab']
    audio_ext = ['.mp3', '.wav', '.flac']
    song_folders = os.listdir(util.BEATLES_SONG)
    chord_folders = os.listdir(util.BEATLES_CHORD)
    chromagrams = []
    annotated_chromagram = []
    label_list = []
    song_names = []
    chord_name = []
    err = []
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
            chromagram, beat_chroma, beat_frames, beat_t, sr = ish_chroma.chroma(os.path.join(util.BEATLES_SONG, song_f, song_files[i]), hop_length=hopsize, type_=type_, tol=tol)
            try:
                anno_chromas, labels = map_chroma(chromagram, sr, hopsize, chord_lab)
            except AssertionError:
                print('ASSERTION FAILED: {}'.format(os.path.join(util.BEATLES_SONG, song_f, song_files[i])))
                err.append(os.path.join(song_f, song_files[i]))
                continue
            chromagrams.append(chromagram.tolist())
            annotated_chromagram.append(anno_chromas)
            label_list.append(labels)
            song_names.append(os.path.join(song_f, song_files[i]))
            chord_name.append(os.path.join(song_f, chord_files[i]))
    res['chromagram'] = chromagrams
    res['annotated_chromas'] = annotated_chromagram
    res['labels'] = label_list
    res['song_names'] = song_names
    res['chord_name'] = chord_name
    res['err'] = err
    return res

def save_json(file_name, json_dict):
    with open(os.path.join(util.PROCESSED_DATA, file_name), 'w') as fp:
        json.dump(jsonify(json_dict), fp)

def load_beatles(file_name):
    with open(os.path.join(util.PROCESSED_DATA, file_name), 'r') as fp:
        res = json.load(fp)
        chromagram = []
        # convert this back to a dictionary of numpy arrays
        for c in res['chromagram']:
            chromagram.append(np.array(c))
        annotated = []
        for n in res['annotated_chromas']:
            tmp = []
            for i in n:
                tmp.append(np.array(i))
            annotated.append(tmp)
        res['chromagram'] = chromagram
        res['annotated_chromas'] = annotated
        return res

def assert_load(res):
    # checks to make sure the data was loaded correctly
    assert(len(res['chromagram']) == len(res['annotated_chromas']))
    assert(len(res['labels']) == len(res['chromagram']))
    for i in range(len(res['labels'])):
        chromagram = res['chromagram'][i]
        annotated = res['annotated_chromas'][i]
        labels = res['labels'][i]
        try:
            assert(len(labels) == len(annotated)) # assert # of annotated arrays == # of labels
        except AssertionError:
            print(i)
        try:
            sum_val = sum([n.shape[1] for n in annotated])
            assert(sum_val <= chromagram.shape[1] * 1.01 and sum_val >= chromagram.shape[1]* 0.99) # assert total number of chromagram frames in annotated chromagram
        except AssertionError:
            print(i)
            print('ASSERTION')


def remove_song_without_ext(files, ext):
    res = []
    for f in files:
        for e in ext:
            if f.endswith(e):
                res.append(f)
    res.sort()
    return res

def test(album, song_title, chord_title):
    ipdb.set_trace()
    song_dir = util.BEATLES_SONG
    chord_dir = util.BEATLES_CHORD
    chord_ext = '.lab'
    hopsize = 512

    song = song_dir + '/' + album + '/' + song_title # should include 'album' but my file structure is not organized that way yet
    chords = chord_dir + '/' + album + '/'+ chord_title + chord_ext # like this

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


def load_data(type_='cqt_512'):
    if(type_ == 'stft'):
        file_name = 'beatle_data_stft.json'
    elif(type_ == 'cqt_1024'):
        file_name = 'beatle_data_cqt_1024_hop.json'
    elif(type_ == 'cqt_512_hop_2_tol'):
        file_name = 'beatle_data_cqt_512_hop_2_tol.json'
    elif(type_ == 'cqt_512'):
        file_name = 'beatle_data_cqt_512.json'
    else:
        raise Exception('No type_ defined')
    res = load_beatles(file_name)
    assert_load(res)
    return res

def compare_song_chroma(song_folder, song_title):
    cqt_chromagram, _, _, _, _ = ish_chroma.chroma(os.path.join(util.BEATLES_SONG, song_folder, song_title), 'cqt')
    stft_chromagram, _, _, _, _ = ish_chroma.chroma(os.path.join(util.BEATLES_SONG, song_folder, song_title), 'stft')
    ish_chroma.compare_cqt_stft(cqt_chromagram, stft_chromagram)



def run_model_on_beatles(train, model_name, data_independent=False):
    files = ['cqt_512', 'stft', 'cqt_512_hop_2_tol', 'cqt_1024']
    for f in files:
        type_ = ''
        if 'cqt' in f:
            type_ = 'CQT'
        else:
            type_ = 'STFT'
        hop_length = 512
        if '1024' in f:
            hop_length = 1024
        base_name = '{}_{}_{}'.format(model_name, type_, hop_length)
        title = '{} w/ {} Chromagram \nHop Length = {}'.format(model_name, type_, hop_length)
        if '_tol' in f:
            idx = f.index('_tol')
            base_name = base_name + '_' + str(f[idx-1]) + '_tol'
            title = title + ' 0.{} Tolerance'.format(f[idx-1])
        chromagram_data = load_data(f)
        del chromagram_data['err']
        if not data_independent:
            test_data, train_data = util.split_data(chromagram_data, 0.15)
        else:
            test_data = chromagram_data
            train_data = chromagram_data
        model = train(chromagram_data=train_data)
        evaluation = util.evaluate(model, test_data)
        util.save_result('{}.json'.format(base_name), evaluation)
        print(evaluation)
        util.display_err_matrix(matrix=evaluation['err_matrix'],
                                title=title,
                                file_name='{}.png'.format(base_name))


if __name__ == "__main__":
    album = "10CD1_-_The_Beatles"
    song_title = "05 - Wild Honey Pie.flac"
#     chord_title = "CD1_-_05_-_Wild_Honey_Pie"
#     test(album, song_title, chord_title)

#     jsonify(res)
#     save_json(file_name, res)
#     compare_song_chroma(album, song_title)
#     ipdb.set_trace()
#     res = map_beatles_dataset(type_='cqt', tol=0.2)
    file_name = 'beatle_data_cqt_512_hop_2_tol.json'
    # save_json(file_name, res)

    res = load_beatles(file_name)
    assert_load(res)





