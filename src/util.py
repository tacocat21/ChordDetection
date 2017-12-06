import random
import collections
import os
import numpy as np
import ipdb
CHORDS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B', 'N'] # N is for no chord
CHORD_IDX = {}
for idx, chord in enumerate(CHORDS):
    CHORD_IDX[chord] = idx
FLAT_EQUIL = {'Db': 'C#', 'Eb':'D#', 'Gb':'F#', 'Ab':'G#', 'Bb':'A#'}
NUM_CHORDS = len(CHORDS)
CHROMAGRAM_SIZE = 12
SRC_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(SRC_DIR, '..', 'data')
BEATLES_DIR = os.path.join(SRC_DIR, '..', 'data', 'The_Beatles_Annotations')
BEATLES_CHORD = os.path.join(BEATLES_DIR, 'chordlab', 'The_Beatles')
BEATLES_SONG = os.path.join(BEATLES_DIR, 'song')
BEATLES_DATA_JSON = os.path.join(DATA_DIR, 'beatle_data.json')

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
            if (prediction[i] == stretched_label[i]):
                curr_song_correct += 1
                total_correct_frames += 1
#        total_correct_frames += curr_song_correct
        print(curr_song_correct/num_frames)
        arco_sum += curr_song_correct/num_frames
    result = {}
    print("Correct: {}/{} = {}".format(total_correct_frames, total_frames, total_correct_frames/total_frames))
    result['TRCO'] = total_correct_frames/total_frames
    result['ARCO'] = arco_sum/num_songs
    print('TRCO: {}\nARCO: {}'.format(result['TRCO'], result['ARCO']))
    return result

def bucket_sort(labels, annotated_chroma):
    """
    
    :param labels: labels from all of the songs
    :param annotated_chroma: chromagram frames from the songs
    :return: dictionary label -> chromagram frames
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

def get_frames(song):
    # TODO: depends on the way the data is formatted
    pass

def distance(prediction, correct_label):
    #TODO: modify this depending on the way data is formatted
    if prediction == correct_label:
        return 1
    return 0