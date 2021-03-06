import util
import numpy as np
import numpy.linalg as la
import beatles_annotated_chroma

class Template:
    def __init__(self):
        self.weights = [create_templates(i) for i in range(util.CHROMAGRAM_SIZE)]
        self.weights.append([np.array([0.2]*util.CHROMAGRAM_SIZE)])

    def predict_frame(self, frame):
        res = []
        # print(frame)
        for weight in self.weights:
            max_val = -1
            for w in weight:
                calc = np.dot(frame, w)
                max_val = max(calc, max_val)
            res.append(max_val)
        return int(np.argmax(res))

    def predict(self, song):
        # res = []
        return np.apply_along_axis(self.predict_frame, axis=1, arr=song)


def distance(v1, v2):
    return np.dot(v1/la.norm(v1), v2/la.norm(v2))
    # return la.norm(v1-v2, 2)

def create_templates(idx):
    res = []
    # jumps = [[3,3], [3,4], [4,3], [4,4]]
    jumps = [[3, 3], [3, 4], [4, 3], [4, 4],
             [3, 3, 3], [3, 3, 4],  # diminished, half dim
             [3, 4, 3], [3, 4, 4],  # minor, minormajor
             [4, 3, 3], [4, 3, 4],  # dominant, major
             [4, 4, 3],  # augmented
             [2, 5], [5, 2]  # sus2, sus4
             ]
    for jump in jumps:
        z = np.zeros(util.CHROMAGRAM_SIZE)
        curr = idx
        z[idx] = 1
        for i in jump:
            curr+= i
            z[curr% util.CHROMAGRAM_SIZE] = 1
        res.append(z)
    return res

def train(chromagram_data):
    return Template()

if __name__ == '__main__':
    files = ['cqt_8192_complete']
    beatles_annotated_chroma.run_model_on_beatles(train, 'Template_model', files=files, data_independent=True)
