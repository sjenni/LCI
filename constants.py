import os

# Directories for data and logs
DATA_DIR = '/Data'
LOG_DIR = os.path.join(DATA_DIR, 'Logs')
TFREC_DIR = os.path.join(DATA_DIR, 'TF_Records')
DATASETS_DIR = os.path.join(DATA_DIR, 'Datasets')

# TFRecord directories
STL10_TF_DATADIR = os.path.join(TFREC_DIR, 'STL10_TFRecords/')

# Source directories for datasets
STL10_DATADIR = os.path.join(DATASETS_DIR, 'STL10/')
