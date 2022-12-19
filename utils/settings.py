"""
settings.py: Define all constant variables in the project

__author__      = "Phan Minh Khanh"
__date__        = 16/12/2022
__copyright__   = "Copyright 2022, The Person-Reidentification Project"
__license__     = "Apache"
__version__     = "2.0"
__email__       = "khanhpm@gmail.com"
"""

import os
import sys

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(ROOT_DIR, "common/python"))


# List of devices can be used for training/recognizing/detection
DEVICE_KINDS = ["CPU", "GPU", "MYRIAD", "HETERO", "HDDL"]
