from hmmlearn import hmm
import util
import numpy as np
import beatles_annotated_chroma
import ipdb
from sklearn.preprocessing import normalize
import collections

def mean_matrix(sorted_dict):
    """

    :param sorted_dict: dictionary output from util.bucket_sort
    :return: mean matrix of training data. Shape: (classes, chromagram)
    """
    res = []
    for c in util.CHORDS:
        res.append(np.mean(sorted_dict[c], axis=1))
    return np.array(res)

def cov_matrix(sorted_dict):
    """

    :param sorted_dict: dictionary output from util.bucket_sort
    :return: covariance matrix of training data. Shape: (classes, chromagram, chromagram)
    """
    res = []
    for c in util.CHORDS:
        res.append(np.cov(sorted_dict[c]))
    return np.array(res)

def initial_distribution(labels):
    """
        :param chord_data: list of list of chords
        :return: count of the first chord of every song in training data. Shape: (NUM_CHORDS, NUM_CHORDS)
    """
    result = np.zeros(util.NUM_CHORDS)
    for l in labels:
        result[util.CHORD_IDX[util.get_base_chord(l[0])]] += 1
    return np.concatenate(normalize([result], axis=1, norm='l1'))


def transition_matrix(labels, annotated_chroma):
    """
        :param chord_data: list of list of chords
        :return: count of the chord transition of every song in training data. Shape: (NUM_CHORDS, NUM_CHORDS)
    """
    result = np.zeros((util.NUM_CHORDS, util.NUM_CHORDS))
    total = 0
    for i in range(len(labels)):

        label = labels[i]
        print('Analyzing label: {}'.format(label))
        ipdb.set_trace()
        prev = util.get_base_chord(label[0])
        prev_idx = util.CHORD_IDX[prev]
        result[prev_idx, prev_idx] += annotated_chroma[i][0].shape[1]
        for l_idx in range(1, len(label)):
            curr_label = util.get_base_chord(label[l_idx])
            prev_idx = util.CHORD_IDX[prev]
            curr_idx = util.CHORD_IDX[curr_label]
            result[prev_idx, curr_idx] += 1
            result[curr_idx, curr_idx] += annotated_chroma[i][l_idx].shape[1] # TODO count this or not?
            total += annotated_chroma[i][l_idx].shape[1]
            prev = curr_label
    return normalize(result, axis=1, norm='l1')

def train(chromagram_data):
    sorted_label = util.bucket_sort(chromagram_data['labels'], chromagram_data['annotated_chromas'])
    #total = [np.concatenate(c, axis=1) for c in chromagram_data['annotated_chromas']]
    # ipdb.set_trace()
    model = hmm.GaussianHMM(n_components=util.NUM_CHORDS, covariance_type='full')
    model.means_ = mean_matrix(sorted_label)
    model.covars_ = cov_matrix(sorted_label)
    model.startprob_ = initial_distribution(chromagram_data['labels'])
    model.transmat_ = transition_matrix(chromagram_data['labels'], chromagram_data['annotated_chromas'])
    # model.fit(total)
    return model

def train_multi(chromagram_data):
    sorted_label = util.bucket_sort(chromagram_data['labels'], chromagram_data['annotated_chromas'])
    total = np.concatenate([np.array(util.match_frame(chromagram_data['labels'][i], chromagram_data['annotated_chromas'][i])).T for i in range(len(chromagram_data['labels']))])
    # total = [[util.CHORD_IDX[util.l] for l in label] for label in chromagram_data['labels']]
    lengths = []
    curr = 0
    for l in chromagram_data['labels']:
        lengths.append(curr)
        curr += len(l)
    #total = np.concatenate(total, axis=1)
    initial_dist = initial_distribution(chromagram_data['labels'])
    trans_matrix = transition_matrix(chromagram_data['labels'], chromagram_data['annotated_chromas'])
    model = hmm.GaussianHMM(n_components=util.NUM_CHORDS)
    model.startprob_ = initial_dist
    model.transmat_ = trans_matrix
    model.emissionprob_ = normalize(mean_matrix(sorted_label), axis=1)
    model.fit([total.T], lengths)
    return model


if __name__ == "__main__":
    # TODO: safe to assume Bb chord => A#chord

    chromagram_data = beatles_annotated_chroma.load_data()
    del chromagram_data['err']
    # ipdb.set_trace()
    # unique_labels = util.count_unique_labels(chromagram_data['labels'])
    # print(unique_labels)
    # for c in unique_labels:
    #     print('{} {}'.format(c, util.get_base_chord(c)))
    # print(len(unique_labels))
    test_data, train_data = util.split_data(chromagram_data, 0.1)
    model = train(chromagram_data=train_data)
    # multi_model = train_multi(chromagram_data)
    evaluation = util.evaluate(model, test_data)
    # print(evaluation)
