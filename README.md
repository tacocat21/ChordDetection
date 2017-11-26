**Setup**

Download the zip files from Google Drive, and unzip them in the data directory.

To obtain the chromagram information, import the function `load_chroma_dict()` from the file `src/load_chromagrams.py`. The function returns an array of dictionaries of the all the songs.

The dictionary will have the following keys:
  * `line`: Shows the line number of the song in the `song_list.txt` file (Not on Github).
  * `title`: The YouTube query used for finding the song and name of the song file.
  * `query`: An HTML encoded version of the original song title.
  * `chromagram`: The song's chromagram in linear time.
  * `beat_chromagram`: The song's chromagram in beat time.
  * `id`: The song's youtube id.
