import os
import pathlib

import county_classifier

PACKAGE_ROOT = pathlib.Path(county_classifier.__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / 'weights'
DATASET_DIR = PACKAGE_ROOT / 'datasets'