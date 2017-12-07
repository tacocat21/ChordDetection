import util
import numpy as np
import beatles_annotated_chroma

class Template:
    def __init__(self):
        self.weights = [create_templates(i) for i in range(util.CHROMAGRAM_SIZE)]
        self.weights.append([np.zeros(util.CHROMAGRAM_SIZE)])

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
    return np.dot(v1, v2)

def create_templates(idx):
    res = []
    jumps = [[3,3], [3,4], [4,3], [4,4]]
    for i, j in jumps:
        z = np.zeros(util.CHROMAGRAM_SIZE)
        z[idx] = 1
        z[(idx + i)% util.CHROMAGRAM_SIZE] = 1
        z[(idx + i + j) % util.CHROMAGRAM_SIZE] = 1
        res.append(z)
    return res

if __name__ == '__main__':
    import ipdb
    ipdb.set_trace()
    data = beatles_annotated_chroma.load_data()
    model = Template()
    util.evaluate(model, data)
