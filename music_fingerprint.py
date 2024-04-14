"""
Finding fingerprint and calculating simple fuzzy similarity 
@author yohanes.gultom@gmail.com
Prerequisites on Ubuntu:
* Python 2.7 and pip
* FFMPEG `sudo apt install ffmpeg`
* AcoustID fingerprinter `sudo apt install acoustid-fingerprinter`
* PyAcoustID `pip install pyacoustid`
* FuzzyWuzzy `pip install fuzzywuzzy[speedup]`
"""

import acoustid
import sys
import os
import chromaprint
import numpy as np
import matplotlib.pyplot as plt
from fuzzywuzzy import fuzz

DIR_DATABASE = 'music/full'
DIR_SAMPLES = 'music/partial'

def get_fingerprint(filepath):
    """
    Get fingerprint (list of signed integer), version, duration
    """
    duration, fp_encoded = acoustid.fingerprint_file(filepath)
    fp, version = chromaprint.decode_fingerprint(fp_encoded)
    return fp, version, duration


def build_fingerprint_database(dirpath, file_ext='.mp3'):
    """
    Build database from directory of audio files
    """
    database = {}
    print('Processing {}..'.format(dirpath))
    for f in os.listdir(dirpath):
        path = os.path.join(dirpath, f)
        name, ext = os.path.splitext(f)
        if os.path.isfile(path) and ext == file_ext:
            print('Getting fingerprint from database item: {}..'.format(f))
            database[f], version, duration = get_fingerprint(path)
    return database


def plot_fingerprints(db):
    """
    Visualize fingerprints in database
    """
    fig = plt.figure()
    numrows = len(db)
    plot_id = 1
    for name, fp in db.iteritems():
        # single column grid
        a = fig.add_subplot(numrows, 1, plot_id)
        imgplot = plt.imshow(get_fingerprint_bitmap(fp))
        a.set_title(name)
        plot_id += 1
    plt.show()


def get_fingerprint_bitmap(fp):
    """
    Plot list of uint32 as (32, len(list)) bitmap
    """
    bitmap = np.transpose(np.array([[b == '1' for b in list('{:32b}'.format(i & 0xffffffff))] for i in fp]))
    return bitmap


if __name__ == '__main__':
    
    # load database and samples
    database = build_fingerprint_database(DIR_DATABASE)
    samples = build_fingerprint_database(DIR_SAMPLES)
    print('\n')

    # find best match of each samples in database
    for sample, sample_fp in samples.iteritems():
        print('Similarity score of "{}":'.format(sample))
        best_match = None
        for name, fp in database.iteritems():
            similarity = fuzz.ratio(sample_fp, fp)
            if not best_match or best_match['score'] < similarity:
                best_match = {
                    'score': similarity,
                    'name': name
                }
            print('{} {}%'.format(name, similarity))
        print('Best match: {name} ({score}%)\n'.format(**best_match))

    # plot database
    plot_fingerprints(database)
