# Copyright 1996-2021 OpenDR European Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import numpy as np
import cv2
from PIL import Image, ImageChops
import torchvision.transforms.functional as TF

class RandomShadows(object):
    def __init__(self, p=0.5, high_ratio=(1,2), low_ratio=(0.01, 0.5), left_low_ratio=(0.4,0.6), \
    left_high_ratio=(0,0.2), right_low_ratio=(0.4,0.6), right_high_ratio = (0,0.2)):
        self.p = p
        self.high_ratio = high_ratio
        self.low_ratio = low_ratio
        self.left_low_ratio = left_low_ratio
        self.left_high_ratio = left_high_ratio
        self.right_low_ratio = right_low_ratio
        self.right_high_ratio = right_high_ratio

    @staticmethod
    def process(img, high_ratio, low_ratio, left_low_ratio, left_high_ratio, \
            right_low_ratio, right_high_ratio):

        w, h = img.size
        high_bright_factor = random.uniform(high_ratio[0], high_ratio[1])
        low_bright_factor = random.uniform(low_ratio[0], low_ratio[1])

        left_low_factor = random.uniform(left_low_ratio[0]*h, left_low_ratio[1]*h)
        left_high_factor = random.uniform(left_high_ratio[0]*h, left_high_ratio[1]*h)
        right_low_factor = random.uniform(right_low_ratio[0]*h, right_low_ratio[1]*h)
        right_high_factor = random.uniform(right_high_ratio[0]*h, right_high_ratio[1]*h)

        tl = (0, left_high_factor)
        bl = (0, left_high_factor+left_low_factor)

        tr = (w, right_high_factor)
        br = (w, right_high_factor+right_low_factor)

        contour = np.array([tl, tr, br, bl], dtype=np.int32)

        mask = np.zeros([h, w, 3],np.uint8)
        cv2.fillPoly(mask,[contour],(255,255,255))
        inverted_mask = cv2.bitwise_not(mask)
        # we need to convert this cv2 masks to PIL images
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # we skip the above convertion because our mask is just black and white
        mask_pil = Image.fromarray(mask)
        inverted_mask_pil = Image.fromarray(inverted_mask)

        low_brightness = TF.adjust_brightness(img, low_bright_factor)
        low_brightness_masked = ImageChops.multiply(low_brightness, mask_pil)
        high_brightness = TF.adjust_brightness(img, high_bright_factor)
        high_brightness_masked = ImageChops.multiply(high_brightness, inverted_mask_pil)

        return ImageChops.add(low_brightness_masked, high_brightness_masked)

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            img = self.process(img, self.high_ratio, self.low_ratio, \
            self.left_low_ratio, self.left_high_ratio, self.right_low_ratio, \
            self.right_high_ratio)
            return img
        else:
            return img
