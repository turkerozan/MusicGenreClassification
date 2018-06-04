# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# We need to convert audio files to wav
class ConvertWav:
    #Path folder for Converting files
    CONVERT_DIRECTORY = "./TrainFiles"
    #from which type to wav 
    FILE_TYPE = "au"

class CreateDataset:
    # Path of GTZAN dataset
    DATASET_DIRECTORY = "./TrainFiles/"

    # Sampling rate (Hz)
    SAMPLING_RATE = 22050

    # Frame size (Samples)
    FRAME_SIZE = 2048

    # Hop Size (Samples)
    HOP_SIZE = 512


class Test:
    # Path for test data
    TEST_DATA_PATH = "./TestFiles/"
