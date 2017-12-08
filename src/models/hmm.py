import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import normalize

import beatles_annotated_chroma
import util
from util import mean_matrix, cov_matrix
import ipdb


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
    for i in range(util.NUM_CHORDS):
        result[i][12] = 1
    assert util.CHORDS[12] == 'N'
    total = 0
    for i in range(len(labels)):

        label = labels[i]
        print('Analyzing label: {}'.format(label))
        # ipdb.set_trace()
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
    # ipdb.set_trace()
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
    ipdb.set_trace()
    beatles_annotated_chroma.run_model_on_beatles(train, 'HMM maj&min', data_independent=False)
    # files = ['cqt_512', 'stft', 'cqt_512_hop_2_tol', 'cqt_1024']
    # for f in files:
    #     type_ = ''
    #     if 'cqt' in f:
    #         type_ = 'CQT'
    #     else:
    #         type_ = 'STFT'
    #     hop_length = 512
    #     if '1024' in f:
    #         hop_length = 1024
    #     chromagram_data = beatles_annotated_chroma.load_data(f)
    #     del chromagram_data['err']
    #     # unique_labels = util.count_unique_labels(chromagram_data['labels'])
    #     # print(unique_labels)
    #     # print(len(unique_labels))
    #     test_data, train_data = util.split_data(chromagram_data, 0.15)
    #     model = train(chromagram_data=train_data)
    #     evaluation = util.evaluate(model, test_data)
    #     util.save_result(f + '.json', evaluation)
    #     print(evaluation)
    #     util.display_err_matrix(matrix=evaluation['err_matrix'], title='HMM w/ {} Chromagram Hop Length = {}'.format(type_, hop_length), file_name=f +'.png')
