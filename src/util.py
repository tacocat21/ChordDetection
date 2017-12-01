import random
CHORDS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B', 'N'] # N is for no chord
CHORD_IDX = {}
for idx, chord in enumerate(CHORDS):
    CHORD_IDX[chord] = idx
NUM_CHORDS = len(CHORDS)

def split_data(input_data, test_ratio):
    """
    
    :param input_data: list of data
    :param test_ratio: pct of data going to testing data
    :return: test data, train data 
    """
    random.shuffle(input_data)
    return input_data[:(test_ratio*len(input_data))], input_data[(test_ratio*len(input_data)):]

def evaluate(model, test_data):
    """
    Evaluate songs using ARCO, and TRCO
    pg. 43 matt mcvicar

    :param model: model to evaluate
    :param test_data: 
    :return: 
    """
    total_frames = 0
    total_correct_frames = 0
    arco_sum = 0
    for song in test_data:
        frames = get_frames(song)
        num_frames = len(frames)
        total_frames += num_frames
        curr_song_correct = 0
        for frame, label in frames:
            res = model.evaluate(frame)
            total_correct_frames += distance(res, label)
            curr_song_correct += distance(res, label)
        arco_sum += curr_song_correct/num_frames
    result = {}
    result['TRCO'] = total_correct_frames/total_frames
    result['ARCO'] = arco_sum/len(test_data)
    print('TRCO: {}\nARCO: {}'.format(result['TRCO'], result['ARCO']))
    return result

def get_frames(song):
    # TODO: depends on the way the data is formatted
    pass

def distance(prediction, correct_label):
    #TODO: modify this depending on the way data is formatted
    if prediction == correct_label:
        return 1
    return 0