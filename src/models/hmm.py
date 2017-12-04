from hmmlearn import hmm
import util
import numpy as np

def mean_matrix(chromagram_data):
    """

    :param chromagram_data: np.matrix with song x chromagram x frame
    :return: mean matrix of training data. Shape: (chromagram,)
    """
    return np.mean(chromagram_data, axis=0)

def cov_matrix(chromagram_data):
    """

    :param chromagram_data: np.matrix with song x chromagram x frame
    :return: covariance matrix of training data. Shape: (chromagram, chromagram)
    """
    return np.cov(chromagram_data)

def initial_distribution(chord_data):
    """
        :param chord_data: list of list of chords
        :return: count of the first chord of every song in training data. Shape: (NUM_CHORDS, NUM_CHORDS)
    """
    result = np.zeros(util.NUM_CHORDS)
    for song in chord_data:
        result[util.CHORD_IDX[song[0]]] += 1
    return result


def transition_matrix(chord_data):
    """
        :param chord_data: list of list of chords
        :return: count of the chord transition of every song in training data. Shape: (NUM_CHORDS, NUM_CHORDS)
    """
    result = np.zeros((util.NUM_CHORDS, util.NUM_CHORDS))
    for song in chord_data:
        prev = 'N'
        for c in song:
            result[util.CHORD_IDX[prev], util.CHORD_IDX[c]] += 1
            prev = c
    return result

def train(chord_data, chromagram_data):
    model = hmm.GaussianHMM(n_components=util.NUM_CHORDS, covariance_type='full')
    model.means_ = mean_matrix(chromagram_data)
    model.covars_ = cov_matrix(chromagram_data)
    model.startprob_ = initial_distribution(chord_data)
    model.transmat_ = transition_matrix(chord_data)
    return model

if __name__ == "__main__":
    #model = train()
    model = hmm.GaussianHMM(n_components=util.NUM_CHORDS, covariance_type='full')
