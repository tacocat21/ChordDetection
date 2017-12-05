from hmmlearn import hmm
import util
import numpy as np
import beatles_annotated_chroma
import ipdb
from sklearn.preprocessing import normalize
import collections

def mean_matrix(labels, annotated_chroma):
    """

    :param chromagram_data: np.matrix with song x chromagram x frame
    :return: mean matrix of training data. Shape: (chromagram,)
    """
    count_dict = collections.defaultdict(list)
    for i in range(len(labels)):
        label = labels[i]
        annotation = annotated_chroma[i]
        assert(len(label) == len(annotation))
        for l_idx in range(len(label)):
            curr_label = util.get_base_chord(label[l_idx])
            count_dict[curr_label].append(annotation[l_idx])
    res = []
    for c in util.CHORDS:
        num_frames = sum([a.shape[1] for a in count_dict[c]])
        total = np.zeros(util.CHROMAGRAM_SIZE)
        for frames in count_dict[c]:
            total+= frames.sum(axis=1)
        res.append(total/num_frames)
    return np.array(res)
def cov_matrix(chromagram_data):
    """

    :param chromagram_data: np.matrix with song x chromagram x frame
    :return: covariance matrix of training data. Shape: (chromagram, chromagram)
    """
    return np.cov(chromagram_data)

def initial_distribution(labels):
    """
        :param chord_data: list of list of chords
        :return: count of the first chord of every song in training data. Shape: (NUM_CHORDS, NUM_CHORDS)
    """
    result = np.zeros(util.NUM_CHORDS)
    for l in labels:
        result[util.CHORD_IDX[util.get_base_chord(l[0])]] += 1
    return result/len(labels)


def transition_matrix(labels, annotated_chroma):
    """
        :param chord_data: list of list of chords
        :return: count of the chord transition of every song in training data. Shape: (NUM_CHORDS, NUM_CHORDS)
    """
    result = np.zeros((util.NUM_CHORDS, util.NUM_CHORDS))
    total = 0
    for i in range(len(labels)):
        label = labels[i]
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
    model = hmm.GaussianHMM(n_components=util.NUM_CHORDS, covariance_type='full')
    model.means_ = mean_matrix(chromagram_data)
    model.covars_ = cov_matrix(chromagram_data)
    model.startprob_ = initial_distribution(chromagram_data['labels'])
    model.transmat_ = transition_matrix(chromagram_data)
    return model

if __name__ == "__main__":
    ipdb.set_trace()
    chromagram_data = beatles_annotated_chroma.load_data()
    mean = mean_matrix(chromagram_data['labels'], chromagram_data['annotated_chromas'])
    cov = cov_matrix()
    initial = initial_distribution(chromagram_data['labels'])
    transition = transition_matrix(chromagram_data['labels'], chromagram_data['annotated_chromas'])
    print(transition)
    print(initial)
