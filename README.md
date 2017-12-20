**Chromagram Setup**

Download the zip files from Google Drive, and unzip them in the data directory.

Link:
<https://drive.google.com/drive/folders/1Jt3vhMknBoUGLvBTnZqpBuKuUSuM56zq?usp=sharing>

To obtain the chromagram information, import the function `load_chroma_dict()` from the file `src/load_chromagrams.py`. The function returns an array of dictionaries of the all the songs.

Each dictionary will have the following keys:
  * `line`: Shows the line number of the song in the `song_list.txt` file (Not on Github).
  * `title`: Name of the song search result in YouTube, used for naming the corresponding wav file.
  * `query`: An HTML encoded version of the original song title and used as a query parameter.
  * `chromagram`: The song's chromagram in linear time.
  * `beat_chromagram`: The song's chromagram in beat time.
  * `id`: The song's youtube id.
  * `annotations`: The song's annotations. Contains the contents of its corresponding annotation
    file in one string.

The beatles dataset was not added in this repository because of copyright issues. 

References:
http://www.mattmcvicar.com/wp-content/uploads/2014/02/thesis.pdf
https://github.com/tmc323/Chord-Annotations
http://isophonics.net/content/reference-annotations-beatles
https://web.stanford.edu/~jurafsky/slp3/9.pdf



