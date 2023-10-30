import json
import os
import numpy as np
from argparse import ArgumentParser

resolution_level = 1
imag_W = 1600
imag_H = 1200

W = imag_W // resolution_level
H = imag_H // resolution_level

camera_setting = {}
# camera_setting['camera_setting'] = []
for i in range(60):
    camera_setting[f'{i}_M'] = np.array([ [1.0, 0.0, 0.0,  0.0],
                            [0.0, 1.0, 0.0,  0.0],
                            [0.0, 0.0, 1.0, -3.0],
                            [0.0, 0.0, 0.0,  1.0] ], dtype=np.float32).tolist()
    camera_setting[f'{i}_K'] = np.array( [
                                [2900, 0, W/2],
                                [0, 2900, H/2],
                                [0, 0, 1]
                            ], dtype=np.float32).tolist()

with open('dynamic_test/cameras_setting.json', 'w') as f:
    json.dump(camera_setting, f, indent=4)