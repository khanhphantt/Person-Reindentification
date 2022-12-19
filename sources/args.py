#!/usr/bin/env python

"""
args.py: Define the input arguments for this project

__author__      = "Phan Minh Khanh"
__date__        = 16/12/2022
__copyright__   = "Copyright 2022, The Person-Reidentification Project"
__license__     = "Apache"
__version__     = "2.0"
__email__       = "khanhpm@gmail.com"
"""

import os
from argparse import ArgumentParser

from sources.settings import (
    DEVICE_KINDS,
    ROOT_DIR,
)


def build_argparser():
    parser = ArgumentParser()

    parser.add_argument(
        "-i",
        "--input",
        # required=True,
        nargs="+",
        help="Paths to video files",
    )

    parser.add_argument(
        "--m_detector",
        default="models/person-detection-retail-0013/FP16/person-detection-retail-0013.xml",  # noqa: E501
        type=str,
        required=False,
        help="Path to the object detection model",
    )

    parser.add_argument(
        "--m_reid",
        default="models/person-reidentification-retail-0288/FP16/person-reidentification-retail-0288.xml",  # noqa: E501
        type=str,
        help="Path to the object re-identification model",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(ROOT_DIR, "sources/configs/person.py"),
        required=False,
        help="Configuration file",
    )
    parser.add_argument(
        "--loop",
        default=False,
        action="store_true",
        help="Optional. Enable reading the input in a loop",
    )
    parser.add_argument(
        "--t_detector",
        type=float,
        default=0.6,
        help="Threshold for the object detection model",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="CPU",
        choices=DEVICE_KINDS,
        help="Optional. Target device. Default value is CPU.",
    )

    parser.add_argument(
        "--show",
        help="Optional. Don't show output",
        action="store_true",
    )
    parser.add_argument(
        "--output_video",
        type=str,
        default="",
        required=False,
        help="Optional. Path to output video",
    )
    return parser
