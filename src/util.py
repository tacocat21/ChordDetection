import json
import collections
import os
import numpy as np
import matplotlib.pyplot as plt

CHROMAGRAM_BASE = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
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
CHORD_MEAN_COV_DIR = os.path.join(RESULT_DIR, 'chord_mean_cov_images')
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
    test_idx = int((test_ratio)*len(input_data['chromagram']))
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
        arco_sum += curr_song_correct/num_frames
    result = {}
    print("Correct: {}/{} = {}".format(total_correct_frames, total_frames, total_correct_frames/total_frames))
    result['TRCO'] = total_correct_frames/total_frames
    result['ARCO'] = arco_sum/num_songs
    result['err_matrix'] = error_matrix
    print('TRCO: {}\nARCO: {}'.format(result['TRCO'], result['ARCO']))
    return result

def display_err_matrix(matrix, title='', file_name=''):
    """
    Display the error matrix
    :param matrix: matrix to display
    :param title: title of the image
    :param file_name: name of the file to save the error matrix
    :return: None
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix, interpolation='nearest')
    fig.colorbar(cax)
    ax.xaxis.set_ticks(np.arange(0, NUM_CHORDS, 1))
    ax.xaxis.tick_bottom()
    ax.yaxis.set_ticks(np.arange(0, NUM_CHORDS, 1))
    ax.set_xticklabels(CHORDS)
    ax.set_yticklabels(CHORDS)
    ax.set_xlabel('Predicted chord')
    ax.set_ylabel('Expected chord')
    plt.suptitle('Error Matrix Expected vs. Model Prediction for ' + title)
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
    """
    
    :param labels: Labels to count
    :return: set containing all of the unique labels in the data
    """
    unique = set()
    for label in labels:
        unique |= set(label)
    return unique

def save_result(file_name, data):
    """
    
    :param file_name: name of the result file
    :param data: data to save to file
    :return: None
    """
    with open(os.path.join(JSON_RESULT_DIR, file_name), 'w') as fp:
        json.dump(jsonify(data), fp)

def load_result(file_name):
    """
    
    :param file_name: name of the result file
    :return: Loaded file to dictionary
    """
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

def display_mean_cov_for_chord(chord_name, mean, cov, file_name=''):
    fig = plt.figure()
    a1 = plt.subplot2grid((4,4), (0,0), rowspan=1, colspan=3)
    a1c = plt.subplot2grid((4,4), (0,3), rowspan=1, colspan=1)
    ax = plt.subplot2grid((4,4), (1,0), rowspan=3, colspan=3)
    axc = plt.subplot2grid((4,4), (1,3), rowspan=3, colspan=1)

    ca1 = a1.matshow(np.array([mean]), interpolation='nearest')
    cbar = plt.colorbar(ca1, cax=a1c)

    a1.xaxis.set_ticks(np.arange(0, CHROMAGRAM_SIZE, 1))
    a1.set_xticklabels(CHROMAGRAM_BASE)
    a1.yaxis.set_ticks_position('none')
    a1.yaxis.set_visible(False)

    cax = ax.matshow(cov, interpolation='nearest')
    cbarx = plt.colorbar(cax, cax=axc)
    ax.xaxis.set_ticks(np.arange(0, CHROMAGRAM_SIZE, 1))
    ax.yaxis.set_ticks(np.arange(0, CHROMAGRAM_SIZE, 1))
    ax.set_xticklabels(CHROMAGRAM_BASE)
    ax.set_yticklabels(CHROMAGRAM_BASE)
    plt.suptitle('Mean and covariance for {} chord'.format(chord_name))
    a1.set_title('Mean Vector')
    ax.set_title('Covariance Matrix', y=0.98)
    a1.xaxis.tick_bottom()
    ax.xaxis.tick_bottom()
    if file_name != '':
        plt.savefig(os.path.join(CHORD_MEAN_COV_DIR, file_name))
    plt.show()

def remove_song(name_list, data):
    name_idx = []
    for name in name_list:
        try:
            name_idx.append(data['song_names'].index(name))
        except ValueError:
            continue
    name_idx.sort(reverse=True)
    for idx in name_idx:
        del data['chromagram'][idx]
        del data['annotated_chromas'][idx]
        del data['labels'][idx]
        del data['song_names'][idx]
        del data['chord_name'][idx]
    return data

def mean_matrix(sorted_dict):
    """

    :param sorted_dict: dictionary output from util.bucket_sort
    :return: mean matrix of training data. Shape: (classes, chromagram)
    """
    res = []
    for c in CHORDS:
        res.append(np.mean(sorted_dict[c], axis=1))
    return np.array(res)


def cov_matrix(sorted_dict):
    """

    :param sorted_dict: dictionary output from util.bucket_sort
    :return: covariance matrix of training data. Shape: (classes, chromagram, chromagram)
    """
    res = []
    for c in CHORDS:
        res.append(np.cov(sorted_dict[c]))
    return np.array(res)

