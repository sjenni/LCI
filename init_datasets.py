import os
from constants import DATA_DIR, DATASETS_DIR, TFREC_DIR
from datasets import download_and_convert_stl10

for dir in [DATA_DIR, DATASETS_DIR, TFREC_DIR]:
    if not os.path.exists(dir):
        os.mkdir(dir)

download_and_convert_stl10.run()
