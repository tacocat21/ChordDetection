from videos_found import video_dict
from glob import glob
import numpy as np
import ipdb
base_data_dir = '../data/'

"""
Returns the base name of the linear and beat chromagram files.
"""
def chroma_filenames(base_file_name):
    chroma_array_dir = base_data_dir + 'chroma_arrays/'
    chroma_extension = '_chroma.npy'
    beat_extension = '_beat.npy'
    chroma_file = chroma_array_dir + base_file_name + chroma_extension
    beat_file = chroma_array_dir + base_file_name + beat_extension
    return chroma_file, beat_file


"""
Loads the chroma and beat array for a specific file.
"""
def load_chroma_array(beat_filename, chroma_filename):
    beat_chromagram = np.load(beat_filename)
    chromagram = np.load(chroma_filename)
    return beat_chromagram, chromagram


"""
Loads the linear and beat chromagram from a file.
"""
def load_chroma_file_from_id(song_id):
    chroma_array = glob(base_data_dir + 'chroma_arrays/*{}*'.format(song_id))
    beat_title = chroma_array[0]
    chroma_title = chroma_array[1]
    assert('beat' in beat_title)
    assert('chroma' in chroma_title)
    return beat_title, chroma_title


"""
Loads a dictionary with the info of all the song files, including the chromogram
of each song. This function should be imported from another file.
"""
def load_chroma_dict():
    # print('Processing Chroma Dict')
    i = 0
    result = []
    for song_dict in video_dict:
        try:
            # current_title = song_dict['title']
            current_id = song_dict['id']
            beat_title, chroma_title = load_chroma_file_from_id(current_id)
            # beat_chromagram, chromagram = load_chroma_array(current_title)
            beat_chromagram, chromagram = load_chroma_array(beat_title, chroma_title)
            ipdb.set_trace()
            song_dict['chromagram'] = chromagram
            song_dict['beat_chromagram'] = beat_chromagram
            result.append(song_dict)
        except AssertionError:
            print("AssertionError occured for {}".format(str(song_dict)))
            continue
    return result

def load_chromagram_tensor():
    result = []
    for song_dict in video_dict:
        try:
            # current_title = song_dict['title']
            current_id = song_dict['id']
            beat_title, chroma_title = load_chroma_file_from_id(current_id)
            # beat_chromagram, chromagram = load_chroma_array(current_title)
            beat_chromagram, chromagram = load_chroma_array(beat_title, chroma_title)
            result.append(chromagram)
        except AssertionError:
            print("AssertionError occured for {}".format(str(song_dict)))
            continue

def main():
    print('Running main...')
    ipdb.set_trace()
    load_chroma_dict()

if __name__ == '__main__':
    main()
