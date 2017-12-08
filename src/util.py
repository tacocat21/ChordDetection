import random
import json
import collections
import os
import numpy as np
import ipdb
import matplotlib.pyplot as plt

# CHORDS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B', 'N'] # N is for no chord
# N is for no chord
CHORDS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B', 'N',
          'Cmin', 'C#min', 'Dmin', 'D#min', 'Emin', 'Fmin', 'F#min', 'Gmin', 'G#min', 'Amin', 'A#min', 'Bmin',
          'Csus2', 'C#sus2', 'Dsus2', 'D#sus2', 'Esus2', 'Fsus2', 'F#sus2', 'Gsus2', 'G#sus2', 'Asus2', 'A#sus2', 'Bsus2',
          'Csus4', 'C#sus4', 'Dsus4', 'D#sus4', 'Esus4', 'Fsus4', 'F#sus4', 'Gsus4', 'G#sus4', 'Asus4', 'A#sus4', 'Bsus4']
CHORD_IDX = {}
for idx, chord in enumerate(CHORDS):
    CHORD_IDX[chord] = idx
FLAT_EQUIL = {'Db': 'C#', 'Eb':'D#', 'Gb':'F#', 'Ab':'G#', 'Bb':'A#'}
NUM_CHORDS = len(CHORDS)

CHROMAGRAM_SIZE = 12
SRC_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(SRC_DIR, '..', 'data')
RESULT_DIR = os.path.join(DATA_DIR, 'result')
JSON_RESULT_DIR = os.path.join(RESULT_DIR, 'json')
IMAGE_RESULT_DIR = os.path.join(RESULT_DIR, 'images')
BEATLES_DIR = os.path.join(SRC_DIR, '..', 'data', 'The_Beatles_Annotations')
BEATLES_CHORD = os.path.join(BEATLES_DIR, 'chordlab', 'The_Beatles')
BEATLES_SONG = os.path.join(BEATLES_DIR, 'song')
BEATLES_DATA_JSON = os.path.join(DATA_DIR, 'beatle_data.json')
PROCESSED_DATA = os.path.join(DATA_DIR, 'processed_data')

""" *** Miscellaneous notes on chord classes, from a theory to Beatles Annotations perspective
SUS CLASS:
    sus2, sus4

MINOR CLASS:
    min[7]  - minor triad, minor 7th
    dim[7]  - dim triad, dim 7th
    hdim7   - half dim 7th
    minmaj7 - minor major 7th
    b3      - automatically minor -> min/dim triad, min/dim/half_dim/minmaj 7th all contain b3

MAJOR CLASS:
    maj[7] - major triad, major 7th
    aug[7] - aug triad, aug 7th
    :7  - dom 7th, root pos
    b7  - dom 7th, iii inv (assuming minors have been filtered out already)
    /7  - maj 7th, iii inv
"""

# Programmatic filtration processing of the chord: SUS -> MIN -> MAJ -> UNKNOWN (assume MAJ)
SUS_CLASS = ['sus2', 'sus4']        # sus2, sus4
MIN_CLASS = ['min', 'dim', 'b3']    # 3: min, dim   7: min7, dim7, hdim7, minmaj7
MAJ_CLASS = ['maj', 'aug', '7']     # 3: maj, aug   7: maj7, aug7, dom7

# File for logging sorted output using a set. Store in a list for iteration later when writing
SUS_SET = set()
MIN_SET = set()
MAJ_SET = set()
ERR_SET = set()
SUS_FILE = open(os.path.join(DATA_DIR, 'sus_chords.txt'), 'w')
MIN_FILE = open(os.path.join(DATA_DIR, 'min_chords.txt'), 'w')
MAJ_FILE = open(os.path.join(DATA_DIR, 'maj_chords.txt'), 'w')
ERR_FILE = open(os.path.join(DATA_DIR, 'err_chords.txt'), 'w')
CHORD_SETS = [SUS_SET, MIN_SET, MAJ_SET, ERR_SET]
CHORD_FILES = [SUS_FILE, MIN_FILE, MAJ_FILE, ERR_FILE]

# Called after all songs have been processed and sets populated with all chords seen
def output_chord_classes(chord_set, chord_file):
    """
    :param chord_set: The set to be sorted
    :param chord_file: Destination for set content
    """
    for c in sorted(chord_set):
        chord_file.write(c + '\n')

def get_base_chord(chord):
    """
    
    :param chord: chord string
    :return: base chord i.e. A/2 -> A
    """
    # Check if bare chord
    if not('/' in chord or ':' in chord):
        return FLAT_EQUIL[chord] if chord in FLAT_EQUIL else chord

    # Else chord has special content
    colon = chord.find(':')
    slash = chord.find('/')
    c_split = (chord[:slash], chord[slash:]) if colon == -1 else (chord[:colon], chord[colon:]) # Split by first colon, or by first slash if no colon
    assert len(c_split) == 1 or len(c_split) == 2, "Chord split failed: {}".format(chord)

    # Use info from 'chord_spec' to place chord into enumerated bin 'chord_root[min]', so either C or Cmin, etc
    chord_root, chord_spec = c_split
    if chord_root in FLAT_EQUIL:
        chord_root = FLAT_EQUIL[chord_root]

    if any(quality in chord_spec for quality in SUS_CLASS):
        SUS_SET.add(chord)
        sus_n = SUS_CLASS[0] if SUS_CLASS[0] in chord_spec else SUS_CLASS[1] # append sus2 or sus4
        return chord_root + sus_n

    if any(quality in chord_spec for quality in MIN_CLASS):
        MIN_SET.add(chord)
        return chord_root + 'min'

    if any(quality in chord_spec for quality in MAJ_CLASS):
        MAJ_SET.add(chord)
        return chord_root

    # No specific classification; likely major chord (/2, /4, /5, /6, /9; nearly no flattened notes, no diminished either). Adding to error set as well
    MAJ_SET.add(chord)
    ERR_SET.add(chord)
    return chord_root

def split_data(input_data, test_ratio):
    """
    
    :param input_data: ch
    :param test_ratio: pct of data going to testing data
    :return: test data, train data 
    """
    test = {}
    train = {}
    test_idx = int((test_ratio)*len(input_data['chromagram']))
    for k in input_data:
        test[k] = input_data[k][:test_idx]
        train[k] = input_data[k][test_idx:]
    return test, train

def match_frame(label, annotated):
    """
    match each frame in annotated with a label
    :param label: 
    :param annotated: 
    :return: 
    """
    assert(len(label) == len(annotated))
    res = []
    for i in range(len(label)):
        res.append([CHORD_IDX[get_base_chord(label[i])]]*annotated[i].shape[1])
    assert(sum([len(r) for r in res]) == sum([a.shape[1] for a in annotated]))
    return np.concatenate(res).tolist()

def evaluate(model, test_data):
    """
    Evaluate songs using ARCO, and TRCO
    pg. 43 matt mcvicar

    :param model: model to evaluate
    :param test_data: test data from splitting the chromagram data
    :return: 
    """
    total_frames = 0
    total_correct_frames = 0
    arco_sum = 0
    num_songs = len(test_data['labels'])
    error_matrix = np.zeros((NUM_CHORDS, NUM_CHORDS)) # correct -> prediction
    for i in range(num_songs): # for each song
        label = test_data['labels'][i]
        annotated = test_data['annotated_chromas'][i]
        chromagram = np.concatenate(test_data['annotated_chromas'][i], axis=1)
        stretched_label = match_frame(label, annotated)
        prediction = model.predict(chromagram.T).tolist()
        num_frames = chromagram.shape[1]
        total_frames += num_frames
        curr_song_correct = 0
        for i in range(len(prediction)):
            if (int(prediction[i]) == int(stretched_label[i])):
                curr_song_correct += 1
                total_correct_frames += 1
            else:
                error_matrix[int(stretched_label[i]), int(prediction[i])] += 1
#        total_correct_frames += curr_song_correct
        print(curr_song_correct/num_frames)
        arco_sum += curr_song_correct/num_frames

    # Output chord classification info
    for i in range(len(CHORD_SETS)):
        output_chord_classes(CHORD_SETS[i], CHORD_FILES[i])

    result = {}
    print("Correct: {}/{} = {}".format(total_correct_frames, total_frames, total_correct_frames/total_frames))
    result['TRCO'] = total_correct_frames/total_frames
    result['ARCO'] = arco_sum/num_songs
    result['err_matrix'] = error_matrix
    print('TRCO: {}\nARCO: {}'.format(result['TRCO'], result['ARCO']))
    return result

def display_err_matrix(matrix, title='', file_name=''):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix, interpolation='nearest')
    fig.colorbar(cax)
    ax.xaxis.set_ticks(np.arange(0, NUM_CHORDS, 1))
    ax.yaxis.set_ticks(np.arange(0, NUM_CHORDS, 1))
    ax.set_xticklabels(CHORDS)
    ax.set_yticklabels(CHORDS)
    plt.title('Error Matrix Expected vs. Model Prediction for ' + title)
    if file_name != '':
        plt.savefig(os.path.join(IMAGE_RESULT_DIR, file_name))

    plt.show()



def bucket_sort(labels, annotated_chroma):
    """
    
    :param labels: labels from all of the songs
    :param annotated_chroma: chromagram frames from the songs
    :return: d*ictionary label -> chromagram frames
    """
    count_dict = collections.defaultdict(list)
    for i in range(len(labels)):
        label = labels[i]
        annotation = annotated_chroma[i]
        assert(len(label) == len(annotation))
        for l_idx in range(len(label)):
            curr_label = get_base_chord(label[l_idx])
            count_dict[curr_label].append(annotation[l_idx])
    res = {}
    for k in count_dict:
        res[k] = np.concatenate(count_dict[k], axis=1)
    return res

def count_unique_labels(labels):
    unique = set()
    for label in labels:
        unique |= set(label)
    return unique

def save_result(file_name, data):
    with open(os.path.join(JSON_RESULT_DIR, file_name), 'w') as fp:
        json.dump(jsonify(data), fp)

def load_result(file_name):
    with open(os.path.join(JSON_RESULT_DIR, file_name), 'r') as fp:
        res = json.load(fp)
    res['err_matrix'] = np.array(res['err_matrix'])
    return res

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

def mean_matrix(sorted_dict):
    """

    :param sorted_dict: dictionary output from util.bucket_sort
    :return: mean matrix of training data. Shape: (classes, chromagram)
    """
    res = []
    for c in CHORDS:
        if c in sorted_dict:
            res.append(np.mean(sorted_dict[c], axis=1))
        else:
            res.append(np.array([-100] * CHROMAGRAM_SIZE))
    return np.array(res)


def cov_matrix(sorted_dict):
    """

    :param sorted_dict: dictionary output from util.bucket_sort
    :return: covariance matrix of training data. Shape: (classes, chromagram, chromagram)
    """
    res = []
    for c in CHORDS:
        if c in sorted_dict:
            res.append(np.cov(sorted_dict[c]))
        else:
            res.append(np.eye(CHROMAGRAM_SIZE)) # chord not in sorted_dict
    return np.array(res)
