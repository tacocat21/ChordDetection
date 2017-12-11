import numpy as np
import util
import beatles_annotated_chroma
import math
import ipdb


class Bayes:
    def __init__(self, mean, covariance):
        self.weights = [quadratic_decision(mean[i], covariance[i]) for i in range(mean.shape[0])]
        self.mean = mean
        self.covariance = covariance

    def get_mean(self, class_idx):
        return self.mean[class_idx]

    def get_cov(self, class_idx):
        return self.covariance[class_idx]

    def predict(self, x_arr):
        res = np.zeros(len(x_arr))
        for idx, x in enumerate(x_arr):
            tmp = []
            for Wi, wi, w in self.weights:
                g = (x.T @ Wi @ x) + wi.T @ x + w
                tmp.append(g)
            res[idx] = int(np.argmax(tmp))
        return res

def quadratic_decision(mean, covariance):
    inv_cov = np.linalg.inv(covariance)
    Wi = -0.5* inv_cov
    wi = inv_cov @ mean
    w = -0.5 * mean.T @ inv_cov @ mean - 0.5*math.log(np.linalg.det(covariance), 2) + math.log(0.5, 2)
    return (Wi, wi, w)

def train(chromagram_data):
    sorted_label = util.bucket_sort(chromagram_data['labels'], chromagram_data['annotated_chromas'])
    mean = util.mean_matrix(sorted_label)
    covar = util.cov_matrix(sorted_label)
    bayes = Bayes(mean, covar)
    return bayes

"""
Correct: 66025/152532 = 0.4328599900348779
TRCO: 0.4328599900348779
ARCO: 0.44113839017957357
"""
if __name__ == '__main__':
    # ipdb.set_trace()
    files = ['stft_2_pow', 'stft_1.5_pow']
    beatles_annotated_chroma.run_model_on_beatles(train, 'Bayes', files=files, data_independent=False)