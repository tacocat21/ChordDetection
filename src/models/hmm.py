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
    sorted_label = util.bucket_sort(chromagram_data['labels'], chromagram_data['annotated_chromas'])
    model = hmm.GaussianHMM(n_components=util.NUM_CHORDS, covariance_type='full')
    model.means_ = mean_matrix(sorted_label)
    model.covars_ = cov_matrix(sorted_label)
    model.startprob_ = initial_distribution(chromagram_data['labels'])
    model.transmat_ = transition_matrix(chromagram_data['labels'], chromagram_data['annotated_chromas'])
    return model

if __name__ == "__main__":
    # hoplengths = [8192]
    # for hoplength in hoplengths:
    #     chromagram_data = beatles_annotated_chroma.load_data('stft_0.5_pow_{}'.format(hoplength))
    #     del chromagram_data['err']
    #     chromagram_data = util.remove_song(['10CD1_-_The_Beatles/06 - The Continuing Story of Bungalow Bill.flac', '10CD1_-_The_Beatles/05 - Wild Honey Pie.flac'], chromagram_data)
    #     test_data, train_data = util.split_data(chromagram_data, 0.15)
    #     model = train(chromagram_data=train_data)
    #     evaluation = util.evaluate(model, test_data)
    #     # ipdb.set_trace()
    #     util.save_result('HMM_STFT_0.5_{}'.format(hoplength) + '.json', evaluation)
    #     util.display_err_matrix(matrix=evaluation['err_matrix'],
    #                             title='HMM w/ {} Chromagram Hop Length = {}'.format('CQT', hoplength),
    #                             )
    # ipdb.set_trace()
    # files = ['stft_0.5_pow', 'stft', 'stft_1.5_pow', 'stft_2_pow']
    files = ['cqt_8192_complete']
    beatles_annotated_chroma.run_model_on_beatles(train, 'HMM', chords=True, files=files, data_independent=False)
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
