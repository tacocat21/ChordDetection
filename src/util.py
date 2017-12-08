import random
import json
import collections
import os
import numpy as np
import ipdb
import matplotlib.pyplot as plt

CHORDS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B', 'N'] # N is for no chord
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

def get_base_chord(c):
    """
    
    :param c: chord string
    :return: base chord i.e. A/2 -> A
    """
    if c in CHORD_IDX:
        return c
    if c in FLAT_EQUIL:
        return FLAT_EQUIL[c]
    try:
        colon_idx = c.index(':')
        return get_base_chord(c[:colon_idx])
    except ValueError:
        pass
    try:
        slash_idx = c.index('/')
        return get_base_chord(c[:slash_idx])
    except ValueError:
        pass
    raise Exception('Could not parse base chord {}'.format(c))


def split_data(input_data, test_ratio):
    """
    
    :param input_data: ch
    :param test_ratio: pct of data going to testing data
    :return: test data, train data 
    """
    test = {}
    train = {}
    test_idx = int(test_ratio*len(input_data['chromagram']))
    for k in input_data:
        test[k] = input_data[k][:test_idx]
        train[k] = input_data[k][test_idx:]
    return test, train

def match_frame(label, annotated):
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

    ax.set_xticklabels([''] + CHORDS)
    ax.set_yticklabels([''] + CHORDS)
    plt.title('Error Matrix Expected vs. Model Prediction for ' + title)
    plt.show()
    if file_name != '':
        plt.savefig(os.path.join(IMAGE_RESULT_DIR, file_name))




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
    with open(os.path.join(RESULT_DIR, file_name), 'w') as fp:
        json.dump(jsonify(data), fp)

def load_result(file_name):
    with open(os.path.join(RESULT_DIR, file_name), 'r') as fp:
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