#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 26 18:34:11 2018

@author: ozan
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 20:47:08 2018

@author: ozan
"""
import sys
sys.path.append("/usr/local/lib/python3/dist-packages")
import essentia 
import essentia
import essentia.standard as es
import librosa
import numpy
import pandas
import os
import sklearn
import config


def main():
    samp_rate = config.CreateDataset.SAMPLING_RATE
    frame_size = config.CreateDataset.FRAME_SIZE
    hop_size = config.CreateDataset.HOP_SIZE
    dataset_dir = config.CreateDataset.DATASET_DIRECTORY

    sub_folders = get_subdirectories(dataset_dir)

    labels = []
    is_created = False

    print("Extracting features from audios...")
    for sub_folder in sub_folders:
        print("Folders : ", sub_folders)
        print(".....Working in folder:", sub_folder)
        sample_arrays = get_sample_arrays(dataset_dir, sub_folder, samp_rate)
        for sample_array in sample_arrays:
            row = extract_features(sample_array, samp_rate, frame_size, hop_size)
            if not is_created:
                dataset_numpy = numpy.array(row)
                is_created = True
            elif is_created:
                dataset_numpy = numpy.vstack((dataset_numpy, row))

            labels.append(sub_folder)

    print("Normalizing the data...")
    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))
    dataset_numpy = scaler.fit_transform(dataset_numpy)

    Feature_Names = ['key','BPM','diatonicSTR','chordsChange','chordsNumber','dancebility','beatsLoud','meanZCR', 'stdZCR', 'meanSpecCentroid', 'stdSpecCentroid', 'meanSpecContrast', 'stdSpecContrast',
                     'meanSpecBandwidth', 'stdSpecBandwidth', 'meanSpecRollof', 'stdSpecRollof',
                     'meanMFCC_1', 'stdMFCC_1', 'meanMFCC_2', 'stdMFCC_2', 'meanMFCC_3', 'stdMFCC_3',
                     'meanMFCC_4', 'stdMFCC_4', 'meanMFCC_5', 'stdMFCC_5', 'meanMFCC_6', 'stdMFCC_6',
                     'meanMFCC_7', 'stdMFCC_7', 'meanMFCC_8', 'stdMFCC_8', 'meanMFCC_9', 'stdMFCC_9',
                     'meanMFCC_10', 'stdMFCC_10', 'meanMFCC_11', 'stdMFCC_11', 'meanMFCC_12', 'stdMFCC_12',
                     'meanMFCC_13', 'stdMFCC_13',
                     'meanHPCP_1','meanHPCP_2','meanHPCP_3','meanHPCP_4','meanHPCP_5',
                     'meanHPCP_6','meanHPCP_7','meanHPCP_8','meanHPCP_9','meanHPCP_10',
                     'meanHPCP_11','meanHPCP_12','meanHPCP_13','meanHPCP_14','meanHPCP_15',
                     'meanHPCP_16','meanHPCP_17','meanHPCP_18','meanHPCP_19','meanHPCP_20',
                     'meanHPCP_21','meanHPCP_22','meanHPCP_23','meanHPCP_24','meanHPCP_25',
                     'meanHPCP_26','meanHPCP_27','meanHPCP_28','meanHPCP_29','meanHPCP_30',
                     'meanHPCP_31','meanHPCP_32','meanHPCP_33','meanHPCP_34','meanHPCP_35',
                     'meanHPCP_36',
                     'stdHPCP_1', 'stdHPCP_2', 'stdHPCP_3', 'stdHPCP_4', 'stdHPCP_5',
                      'stdHPCP_6', 'stdHPCP_7', 'stdHPCP_8', 'stdHPCP_9', 'stdHPCP_10',
                       'stdHPCP_11', 'stdHPCP_12', 'stdHPCP_13', 'stdHPCP_14', 'stdHPCP_15',
                        'stdHPCP_16', 'stdHPCP_17', 'stdHPCP_18', 'stdHPCP_19', 'stdHPCP_20',
                         'stdHPCP_21', 'stdHPCP_22', 'stdHPCP_23', 'stdHPCP_24', 'stdHPCP_25',
                          'stdHPCP_26', 'stdHPCP_27', 'stdHPCP_28', 'stdHPCP_29', 'stdHPCP_30',
                           'stdHPCP_31', 'stdHPCP_32', 'stdHPCP_33', 'stdHPCP_34', 'stdHPCP_35',
                            'stdHPCP_36',
                     ]
    dataset_pandas = pandas.DataFrame(dataset_numpy, columns=Feature_Names)

    dataset_pandas["genre"] = labels
    dataset_pandas.to_csv("data_set4.csv", index=False)
    print("Data set has been created and sent to the project folder!")

def get_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def get_sample_arrays(dataset_dir, folder_name, samp_rate):
    path_of_audios = librosa.util.find_files(dataset_dir + "/" + folder_name)
    return path_of_audios


def extract_features(path, sample_rate, frame_size, hop_size):
    signal, sr = librosa.load(path, sr=sample_rate, duration=5.0)   
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=signal, frame_length=frame_size, hop_length=hop_size)
    spectral_centroid = librosa.feature.spectral_centroid(y=signal, sr=sample_rate, n_fft=frame_size,
                                                              hop_length=hop_size)
    spectral_contrast = librosa.feature.spectral_contrast(y=signal, sr=sample_rate, n_fft=frame_size,
                                                              hop_length=hop_size)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sample_rate, n_fft=frame_size,
                                                                hop_length=hop_size)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sample_rate, n_fft=frame_size, hop_length=hop_size)
    mfccs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_fft=frame_size, hop_length=hop_size)
    features, features_frames = es.MusicExtractor(lowlevelStats=['mean', 'stdev'], 
                                              rhythmStats=['mean', 'stdev'], 
                                              tonalStats=['mean', 'stdev'])(path)
    bpm = features['rhythm.bpm']
    key = features['tonal.key_temperley.scale']
    keyout = 0
    if key == 'minor':
        keyout = 0
    else:
        keyout = 1
    
    print(path + ' \n' )
    print("BPM:", features['rhythm.bpm'])
    print("Key: ", key )
    print(" keyout : ", keyout)
    return [
        keyout,
        bpm,
        features['tonal.tuning_diatonic_strength'],
        features['tonal.chords_changes_rate'],
        features['tonal.chords_number_rate'],
        features['rhythm.danceability'],
        features['rhythm.beats_loudness.mean'],
        numpy.mean(zero_crossing_rate),
        numpy.std(zero_crossing_rate),
        numpy.mean(spectral_centroid),
        numpy.std(spectral_centroid),
        numpy.mean(spectral_contrast),
        numpy.std(spectral_contrast),
        numpy.mean(spectral_bandwidth),
        numpy.std(spectral_bandwidth),
        numpy.mean(spectral_rolloff),
        numpy.std(spectral_rolloff), 
        numpy.mean(mfccs[1, :]),
        numpy.std(mfccs[1, :]),
        numpy.mean(mfccs[2, :]),
        numpy.std(mfccs[2, :]),
        numpy.mean(mfccs[3, :]),
        numpy.std(mfccs[3, :]),
        numpy.mean(mfccs[4, :]),
        numpy.std(mfccs[4, :]),
        numpy.mean(mfccs[5, :]),
        numpy.std(mfccs[5, :]),
        numpy.mean(mfccs[6, :]),
        numpy.std(mfccs[6, :]),
        numpy.mean(mfccs[7, :]),
        numpy.std(mfccs[7, :]),
        numpy.mean(mfccs[8, :]),
        numpy.std(mfccs[8, :]),
        numpy.mean(mfccs[9, :]),
        numpy.std(mfccs[9, :]),
        numpy.mean(mfccs[10, :]),
        numpy.std(mfccs[10, :]),
        numpy.mean(mfccs[11, :]),
        numpy.std(mfccs[11, :]),
        numpy.mean(mfccs[12, :]),
        numpy.std(mfccs[12, :]),
        numpy.mean(mfccs[13, :]),
        numpy.std(mfccs[13, :]),
        features['tonal.hpcp.mean'][0],
        features['tonal.hpcp.mean'][1],
        features['tonal.hpcp.mean'][2],
        features['tonal.hpcp.mean'][3],
        features['tonal.hpcp.mean'][4],
        features['tonal.hpcp.mean'][5],
        features['tonal.hpcp.mean'][6],
        features['tonal.hpcp.mean'][7],
        features['tonal.hpcp.mean'][8],
        features['tonal.hpcp.mean'][9],
        features['tonal.hpcp.mean'][10],
        features['tonal.hpcp.mean'][11],
        features['tonal.hpcp.mean'][12],
        features['tonal.hpcp.mean'][13],
        features['tonal.hpcp.mean'][14],
        features['tonal.hpcp.mean'][15],
        features['tonal.hpcp.mean'][16],
        features['tonal.hpcp.mean'][17],
        features['tonal.hpcp.mean'][18],
        features['tonal.hpcp.mean'][19],
        features['tonal.hpcp.mean'][20],
        features['tonal.hpcp.mean'][21],
        features['tonal.hpcp.mean'][22],
        features['tonal.hpcp.mean'][23],
        features['tonal.hpcp.mean'][24],
        features['tonal.hpcp.mean'][25],
        features['tonal.hpcp.mean'][26],
        features['tonal.hpcp.mean'][27],features['tonal.hpcp.mean'][28],
        features['tonal.hpcp.mean'][29],
        features['tonal.hpcp.mean'][30],
        features['tonal.hpcp.mean'][31],
        features['tonal.hpcp.mean'][32],
        features['tonal.hpcp.mean'][33],
        features['tonal.hpcp.mean'][34],
        features['tonal.hpcp.mean'][35],
        features['tonal.hpcp.stdev'][0],
        features['tonal.hpcp.stdev'][1],
        features['tonal.hpcp.stdev'][2],
        features['tonal.hpcp.stdev'][3],
        features['tonal.hpcp.stdev'][4],features['tonal.hpcp.stdev'][5],
        features['tonal.hpcp.stdev'][6],features['tonal.hpcp.stdev'][7],
        features['tonal.hpcp.stdev'][8],features['tonal.hpcp.stdev'][9],
        features['tonal.hpcp.stdev'][10],features['tonal.hpcp.stdev'][11],
        features['tonal.hpcp.stdev'][12],
        features['tonal.hpcp.stdev'][13],
        features['tonal.hpcp.stdev'][14],
        features['tonal.hpcp.stdev'][15],features['tonal.hpcp.stdev'][16],
        features['tonal.hpcp.stdev'][17],features['tonal.hpcp.stdev'][18],
        features['tonal.hpcp.stdev'][19],features['tonal.hpcp.stdev'][20],
        features['tonal.hpcp.stdev'][21],features['tonal.hpcp.stdev'][22],
        features['tonal.hpcp.stdev'][23],features['tonal.hpcp.stdev'][24],
        features['tonal.hpcp.stdev'][25],features['tonal.hpcp.stdev'][26],
        features['tonal.hpcp.stdev'][27],features['tonal.hpcp.stdev'][28],
        features['tonal.hpcp.stdev'][29],features['tonal.hpcp.stdev'][30],
        features['tonal.hpcp.stdev'][31],features['tonal.hpcp.stdev'][32],
        features['tonal.hpcp.stdev'][33],features['tonal.hpcp.stdev'][34],
        features['tonal.hpcp.stdev'][35],
    ]

main()