import argparse

import cv2
import numpy as np

from openvino.inference_engine import IENetwork, IECore

def main():
    '''
    Start the OpenVINO Sneeze-Cough detector.
    '''
    args = get_args()
    model = ADAS_MODEL
    infer_on_video(args, model)

if __name__ == "__main__":
    main()
